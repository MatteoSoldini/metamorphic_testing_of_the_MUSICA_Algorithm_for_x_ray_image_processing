#include "include/app.h"

int main() {
    App app = App();

    app.init();
    app.run();
    app.cleanup();

    return 0;
}