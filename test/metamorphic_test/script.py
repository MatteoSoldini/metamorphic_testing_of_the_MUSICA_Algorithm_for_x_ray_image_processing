import csv
import math
import os
import subprocess
from typing import List
from PIL import Image, ImageDraw, ImageChops, ImageOps
from matplotlib import pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
import pydicom
from dataclasses import dataclass
from scipy.spatial import distance

RAW_IMAGE_SIZE = 3072
INPUT_PATH = "..\\..\\raw_images\\"
OUTPUT_PATH = "out\\"
EXE_WORK_DIR = "..\\..\\out\\build\\x64-release"
EXE_FILE = "maverick-standalone.exe"
R_CSV_FILE = "direct_robustness.csv"
NR_CSV_FILE = "reg_based_robustness.csv"
S_CSV_FILE = "ref_similarities.csv"

PROCESSING_MARGIN = 10

def load_image(path: str, width: int, height: int) -> Image:
    with open(path, 'rb') as f:
        f.seek(256)
        
        image_data = f.read()
    
    image_array = np.frombuffer(image_data, dtype=np.uint16)
    image_array = image_array.reshape((height, width))
    
    image = Image.fromarray(image_array)
    
    return image

def save_image(path: str, image: Image):
    image_array = np.array(image, dtype=np.uint16)
    image_data = image_array.tobytes()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.seek(256)
        f.write(image_data)
        f.close()

def apply_quantum_noise(image: Image, scale_factor: float = 1.0) -> Image:
    image_np = np.array(image, dtype=np.uint16)

    scaled_image = image_np * scale_factor
    noisy_image_np = np.random.poisson(scaled_image).astype(np.float32)
    noisy_image_np = noisy_image_np / scale_factor

    noisy_image_np = np.clip(noisy_image_np, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    return Image.fromarray(noisy_image_np)

def add_gaussian_noise(image: Image, mean: float, sigma: float):
    image_array = np.array(image, dtype=np.uint16)
    noise = np.random.normal(mean, sigma, image_array.shape).astype(np.int32)
    noisy_image_array = image_array + noise
    noisy_image_array = np.clip(noisy_image_array, 0, 65535).astype(np.uint16)

    return Image.fromarray(noisy_image_array)

def crop(image: Image, crop_area: list[int]) -> Image:
    cropped_image = image.crop(crop_area)
    new_image = Image.new("I;16", image.size, (0))
    paste_position = (crop_area[0], crop_area[1])
    new_image.paste(cropped_image, paste_position)
    return new_image

def apply_collimator(image: Image, shutter_h: int, shutter_v: int) -> Image:
    mask_image = Image.new("1", image.size, (0))
    
    draw = ImageDraw.Draw(mask_image)
    draw.rectangle(
        [
            shutter_h,
            shutter_v,
            image.size[0] - shutter_h,
            image.size[1] - shutter_v
        ],
        fill='white'
    )

    image_array = np.array(image, dtype=np.uint16)
    image_array = image_array / 100
    low_dose_image = Image.fromarray(image_array)
    noisy_image = apply_quantum_noise(low_dose_image, 1)
    
    new_image = Image.composite(image, noisy_image, mask_image)
    return new_image

def clamp_translation(image: Image, x_shift, y_shift=0):
    bright_image_size = 2
    margin = 10

    left = margin if x_shift > 0 else 0
    right = image.width - margin if x_shift < 0 else image.width
    top = margin if y_shift > 0 else 0
    bottom = image.height - margin if y_shift < 0 else image.height
    
    cropped_img = image.crop((left, top, right, bottom))

    b_right = margin + bright_image_size if x_shift > 0 else image.width
    b_bottom = margin + bright_image_size if y_shift > 0 else image.height

    bright_pixel_image = image.crop((left, top, b_right, b_bottom))

    brightest_pixel = int(np.percentile(np.array(bright_pixel_image), 99))
    fill_color = (brightest_pixel,) * len(image.split())
    #brightest_pixel = max(cropped_img.getdata())

    out_image = Image.new(image.mode, image.size, color=fill_color)
    out_image.paste(cropped_img, (x_shift, y_shift))

    return out_image

def clamp_rotate(image: Image, degree):
    margin = 100
    cropped_img = image.crop((
        margin,
        margin,
        image.width - margin,
        image.height - margin,
    ))

    brightest_pixel = int(np.percentile(np.array(cropped_img), 95))
    #brightest_pixel = max(image.getdata())
    #print(brightest_pixel)
    fill_color = (brightest_pixel,) * len(image.split())

    rot_image = cropped_img.rotate(degree, fillcolor=fill_color)

    out_image = Image.new(image.mode, image.size, color=fill_color)
    out_image.paste(rot_image, (margin, margin))

    return out_image

def mse_similarity(image_a: Image, image_b: Image):
    errors = np.asarray(ImageChops.difference(image_a, image_b)) / 255
    return 1.0 - math.sqrt(np.mean(np.square(errors)))

def ssim_similarity(image_a: Image, image_b: Image):
    gray_image1 = np.array(image_a.convert('L'))
    gray_image2 = np.array(image_b.convert('L'))

    ssim_value, _ = ssim(gray_image1, gray_image2, full=True)
    return ssim_value

def hist_similarity(image_a: Image, image_b: Image):
    # image_a.show()
    # image_b.show()

    pixels_a = list(image_a.convert('L').getdata())
    hist_a, bin_edges_a = np.histogram(pixels_a, bins=256)

    pixels_b = list(image_b.convert('L').getdata())
    hist_b, bin_edges_b = np.histogram(pixels_b, bins=256)

    assert len(hist_a) == len(hist_b)

    # # correlation
    # correlation_matrix = np.corrcoef(hist_a, hist_b)
    # correlation = correlation_matrix[0, 1]

    # normalized intersection
    sum = np.sum(np.minimum(hist_a, hist_b))
    intersection = sum / min(np.sum(hist_a), np.sum(hist_b))
    
    # normalized euclidean distance
    sum = 0.0
    for index in range(0, len(hist_a)):
        sum += np.pow(hist_a[index] / np.sum(hist_a) - hist_b[index] / np.sum(hist_b), 2)

    e_distance = np.sqrt(sum) / np.sqrt(2)

    # bhattacharyya distance
    hist_a_n = hist_a / np.sum(hist_a)
    hist_b_n = hist_b / np.sum(hist_b)

    b_distance = 0.0
    for index in range(0, len(hist_a)):
        b_distance += np.sqrt(hist_a_n[index] * hist_b_n[index])

    # plt.hist(pixels_a, bins=256, range=(0, 256), density=True, color='red', alpha=0.5)
    # plt.hist(pixels_b, bins=256, range=(0, 256), density=True, color='blue', alpha=0.5)
    # plt.title('Image Histogram')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.show()

    #print(intersection, e_distance, b_distance)

    return float(intersection), float(e_distance), float(b_distance)

def run_process(raw_file_path, out_file_path):
    raw_file_abs_path = os.path.abspath(raw_file_path)
    out_file_abs_path = os.path.abspath(out_file_path)

    cwd = os.getcwd()
    os.chdir(EXE_WORK_DIR)
    subprocess.run(
        [
            EXE_FILE,
            raw_file_abs_path,
            out_file_abs_path
        ],
        check=True
    )
    os.chdir(cwd)

start = time.time()

print("running")

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

r_csv_file = open(OUTPUT_PATH + R_CSV_FILE, 'w', newline='')
robustness_spamwriter = csv.writer(
    r_csv_file,
    delimiter=',',
    quotechar='"',
    quoting=csv.QUOTE_MINIMAL
)

robustness_spamwriter.writerow([
    'raw file',
    'alteration',
    'altered vs unaltered mse',
    'altered vs unaltered ssim',
    'altered vs unaltered histogram distance',
    'altered vs reference mse',
    'altered vs reference ssim',
    'altered vs reference histogram distance',
    'normalized altered vs reference mse',
    'normalized altered vs reference ssim',
    'normalized altered vs reference histogram distance'
])

def m_sim_alt(
    file_name: str,
    alteration: str,
    alt_image: Image,
    unalt_image: Image,
    ref_image: Image,
    ovd_mse: float,
    ovd_ssim: float,
    ovd_hist: float
):
    own_mse = mse_similarity(alt_image, unalt_image)
    own_ssim = ssim_similarity(alt_image, unalt_image)
    _, own_hist, _ = hist_similarity(alt_image, unalt_image)

    ref_mse = mse_similarity(alt_image, ref_image)
    ref_ssim = ssim_similarity(alt_image, ref_image)
    _, ref_hist, _ = hist_similarity(alt_image, ref_image)

    robustness_spamwriter.writerow([
        file_name,
        alteration,
        own_mse,
        own_ssim,
        own_hist,
        ref_mse,
        ref_ssim,
        ref_hist,
        ref_mse / ovd_mse,
        ref_ssim / ovd_ssim,
        (ref_hist - ovd_hist) / (1.0 - ovd_hist)
    ])

s_csv_file = open(OUTPUT_PATH + S_CSV_FILE, 'w', newline='')
reference_similarities_spamwriter = csv.writer(
    s_csv_file,
    delimiter=',',
    quotechar='"',
    quoting=csv.QUOTE_MINIMAL
)

reference_similarities_spamwriter.writerow([
    'raw file',
    'mse similarity',
    'ssim similarity',
    'histogram distance',
])

def m_sim_ovd(file_name: str, unalt_image: Image, ref_image: Image):
    mse = mse_similarity(unalt_image, ref_image)
    ssim = ssim_similarity(unalt_image, ref_image)
    _, hist, _ = hist_similarity(unalt_image, ref_image)

    reference_similarities_spamwriter.writerow([
        file_name,
        mse,
        ssim,
        hist,
    ])

    return mse, ssim, hist


nr_csv_file = open(OUTPUT_PATH + NR_CSV_FILE, 'w', newline='')
norm_robustness_spamwriter = csv.writer(
    nr_csv_file,
    delimiter=',',
    quotechar='"',
    quoting=csv.QUOTE_MINIMAL
)

norm_robustness_spamwriter.writerow([
    'raw file',
    'alteration',
    'altered vs unaltered mse',
    'altered vs unaltered ssim',
    'altered vs unaltered histogram distance',
    'altered vs reference mse',
    'altered vs reference ssim',
    'altered vs reference histogram distance',
    'normalized altered vs reference mse',
    'normalized altered vs reference ssim',
    'normalized altered vs reference histogram distance'
])

def m_sim_norm_alt(
    file_name: str,
    alteration: str,
    alt_image: Image,
    unalt_image: Image,
    ref_image: Image,
    ovd_mse: float,
    ovd_ssim: float,
    ovd_hist: float
):
    assert alt_image.size == unalt_image.size, "comparing image of different size"
    assert unalt_image.size == ref_image.size, "comparing image of different size"

    # alt_image.show()
    # unalt_image.show()

    # _ = input("wait")

    own_mse = mse_similarity(alt_image, unalt_image)
    own_ssim = ssim_similarity(alt_image, unalt_image)
    _, own_hist, _ = hist_similarity(alt_image, unalt_image)

    ref_mse = mse_similarity(alt_image, ref_image)
    ref_ssim = ssim_similarity(alt_image, ref_image)
    _, ref_hist, _ = hist_similarity(alt_image, ref_image)

    norm_robustness_spamwriter.writerow([
        file_name,
        alteration,
        own_mse,
        own_ssim,
        own_hist,
        ref_mse,
        ref_ssim,
        ref_hist,
        ref_mse / ovd_mse,
        ref_ssim / ovd_ssim,
        (ref_hist - ovd_hist) / (1.0 - ovd_hist)
    ])

@dataclass
class TestFile:
    raw_file: str
    reference_dicom_file: str

test_files: List[TestFile] = [
    TestFile("foot\\image.raw", "foot\\proc"),
    TestFile("hand\\image.raw", "hand\\proc"),
    TestFile("head\\image.raw", "head\\proc"),
    TestFile("knee\\image.raw", "knee\\proc"),
    TestFile("pelvis\\image.raw", "pelvis\\proc"),
    TestFile("thorax\\image.raw", "thorax\\proc"),
]

for test_file in test_files:
    raw_image_name = test_file.raw_file
    raw_file_path = INPUT_PATH + raw_image_name
    unaltered_file_path = OUTPUT_PATH + raw_image_name + "_unaltered.bmp"
    
    os.makedirs(os.path.dirname(OUTPUT_PATH + raw_image_name), exist_ok=True)    
    
    # own
    run_process(raw_file_path, unaltered_file_path)
    raw_image = load_image(raw_file_path, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE)
    unalt_image = Image.open(unaltered_file_path)

    # reference
    ds = pydicom.dcmread(INPUT_PATH + test_file.reference_dicom_file)
    di = Image.fromarray(ds.pixel_array)

    if di.mode == 'I;16':
        di_point = di.point(lambda i: i * (1./256)).convert('L')
        di = di_point.convert('RGB')
    else:
        di = di.convert('RGB')

    reference_image = ImageOps.invert(di)
    
    ovd_mse, ovd_ssim, ovd_hist = m_sim_ovd(
        raw_image_name,
        unalt_image,
        reference_image
    )

    # collimator
    for shutter in range(200, 1000 + 1, 200):
        image = apply_collimator(raw_image, shutter, shutter)
        alteration_name = "c_sh_" + str(shutter)
        altered_file_path = OUTPUT_PATH + raw_image_name + "_" + alteration_name
        save_image(altered_file_path + ".raw", image)

        out_file_path = altered_file_path + ".bmp"
        run_process(altered_file_path + ".raw", altered_file_path + ".bmp")

        alt_image = Image.open(out_file_path)

        # alt_image.show()
        # unalt_image.show()

        # _ = input("wait")

        # measure against own
        m_sim_alt(
            raw_image_name,
            alteration_name,
            alt_image,
            unalt_image,
            reference_image,
            ovd_mse,
            ovd_ssim,
            ovd_hist
        )

        x = shutter + PROCESSING_MARGIN
        y = shutter + PROCESSING_MARGIN
        width = alt_image.width - (2 * shutter + 2 * PROCESSING_MARGIN)
        height = alt_image.height - (2 * shutter + 2 * PROCESSING_MARGIN)

        m_sim_norm_alt(
            raw_image_name,
            alteration_name,
            alt_image.crop((x, y, x + width, y + height)),
            unalt_image.crop((x, y, x + width, y + height)),
            reference_image.crop((x, y, x + width, y + height)),
            ovd_mse,
            ovd_ssim,
            ovd_hist
        )

    # image translate x
    for translate_x in range(300, 1500 + 1, 300):
        image = clamp_translation(raw_image, translate_x, 0)
        #image = raw_image.rotate(0, translate=(translate_x, 0))

        alteration_name = "t_x_" + str(translate_x)
        altered_file_path = OUTPUT_PATH + raw_image_name + "_" + alteration_name
        save_image(altered_file_path + ".raw", image)

        out_file_path = altered_file_path + ".bmp"
        run_process(altered_file_path + ".raw", altered_file_path + ".bmp")
        
        # measure against own
        alt_image = Image.open(out_file_path)

        m_sim_alt(
            raw_image_name,
            alteration_name,
            alt_image,
            unalt_image,
            reference_image,
            ovd_mse,
            ovd_ssim,
            ovd_hist
        )

        m_sim_norm_alt(
            raw_image_name,
            alteration_name,
            alt_image.crop((
                translate_x,
                0,
                alt_image.width,
                alt_image.height
            )),
            unalt_image.crop((
                PROCESSING_MARGIN,
                0,
                alt_image.width - translate_x + PROCESSING_MARGIN,
                alt_image.height
            )),
            reference_image.crop((
                PROCESSING_MARGIN,
                0,
                alt_image.width - translate_x + PROCESSING_MARGIN,
                alt_image.height
            )),
            ovd_mse,
            ovd_ssim,
            ovd_hist
        )

    # image translate y
    for translate_y in range(300, 1500 + 1, 300):
        image = clamp_translation(raw_image, 0, translate_y)
        alteration_name = "t_y_" + str(translate_y)
        altered_file_path = OUTPUT_PATH + raw_image_name + "_" + alteration_name
        save_image(altered_file_path + ".raw", image)

        out_file_path = altered_file_path + ".bmp"
        run_process(altered_file_path + ".raw", altered_file_path + ".bmp")
        
        # measure against own
        alt_image = Image.open(out_file_path)

        m_sim_alt(
            raw_image_name,
            alteration_name,
            alt_image,
            unalt_image,
            reference_image,
            ovd_mse,
            ovd_ssim,
            ovd_hist
        )

        m_sim_norm_alt(
            raw_image_name,
            alteration_name,
            alt_image.crop((
                0,
                translate_y,
                alt_image.width,
                alt_image.height
            )),
            unalt_image.crop((
                0,
                PROCESSING_MARGIN,
                alt_image.width,
                alt_image.height - translate_y + PROCESSING_MARGIN
            )),
            reference_image.crop((
                0,
                PROCESSING_MARGIN,
                alt_image.width,
                alt_image.height - translate_y + PROCESSING_MARGIN
            )),
            ovd_mse,
            ovd_ssim,
            ovd_hist
        )


    # image rotate
    degrees = [9, 18, 27, 36, 45]
    for degree in degrees:
        image = clamp_rotate(raw_image, degree)
        #raw_image.rotate(degree)
        alteration_name = "r_" + str(degree)
        altered_file_path = OUTPUT_PATH + raw_image_name + "_" + alteration_name
        save_image(altered_file_path + ".raw", image)

        out_file_path = altered_file_path + ".bmp"
        run_process(altered_file_path + ".raw", altered_file_path + ".bmp")
        
        alt_image = Image.open(out_file_path)

        m_sim_alt(
            raw_image_name,
            alteration_name,
            alt_image,
            unalt_image,
            reference_image,
            ovd_mse,
            ovd_ssim,
            ovd_hist
        )

        w, h = unalt_image.size        
        angle_rad = math.radians(degree)
        new_w = w * abs(math.cos(angle_rad)) + h * abs(math.sin(angle_rad))
        new_h = h * abs(math.cos(angle_rad)) + w * abs(math.sin(angle_rad))
        
        inner_w = w * h / new_h if w < h else h * w / new_w
        inner_h = h * w / new_w if w < h else w * h / new_h
        
        left = (w - inner_w) / 2
        top = (h - inner_h) / 2
        right = (w + inner_w) / 2
        bottom = (h + inner_h) / 2

        m_sim_norm_alt(
            raw_image_name,
            alteration_name,
            alt_image.crop((left, top, right, bottom)),
            unalt_image.rotate(degree).crop((left, top, right, bottom)),
            reference_image.rotate(degree).crop((left, top, right, bottom)),
            ovd_mse,
            ovd_ssim,
            ovd_hist
        )

    
    # gaussian noise
    sdevs = [4.0, 16.0, 64.0, 256.0, 1024.0]
    for sdev in sdevs:
        image = add_gaussian_noise(raw_image, 0.0, sdev)
        alteration_name = "gn_" + str(sdev)
        altered_file_path = OUTPUT_PATH + raw_image_name + "_" + alteration_name
        save_image(altered_file_path + ".raw", image)

        out_file_path = altered_file_path + ".bmp"
        run_process(altered_file_path + ".raw", altered_file_path + ".bmp")
        
        alt_image = Image.open(out_file_path)

        m_sim_alt(
            raw_image_name,
            alteration_name,
            alt_image,
            unalt_image,
            reference_image,
            ovd_mse,
            ovd_ssim,
            ovd_hist
        )

    # quantum noise
    factors = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    for factor in factors:
        image = apply_quantum_noise(raw_image, factor)
        alteration_name = "pn_" + str(factor)
        altered_file_path = OUTPUT_PATH + raw_image_name + "_" + alteration_name
        save_image(altered_file_path + ".raw", image)

        out_file_path = altered_file_path + ".bmp"
        run_process(altered_file_path + ".raw", altered_file_path + ".bmp")
        
        alt_image = Image.open(out_file_path)

        m_sim_alt(
            raw_image_name,
            alteration_name,
            alt_image,
            unalt_image,
            reference_image,
            ovd_mse,
            ovd_ssim,
            ovd_hist
        )

r_csv_file.close()
s_csv_file.close()
nr_csv_file.close()

end = time.time()
print("testing time:", int((end - start) / 60), "min")
