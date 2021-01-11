#include <iostream>
#include <string>
#include <chrono>

#include "sample.h"

int main(int args, char** argv) {
    string weight_file = "../weights/alexnet.weights";
    string engine_file;
    if (args != 3) {
        cerr << "input arguments as:" << endl;
        cerr << "./example3 -w weights_path -e engine_path" << endl;
        return -1;
    }
    if (string(argv[1]) == "-e") {
        engine_file = string(argv[2]);
    } else {
        cerr << "wrong args: " << string(argv[1]) << endl;
        return -1;
    }

    int b = 1, h = 224, w = 224, o = 1000;
    auto alexnet = SampleAlexnet(b, h, w, o);
    alexnet.build(weight_file, engine_file);

    // prepare resource
    float input[3 * h * w];
    for (int i = 0; i < 3 * h * w; i++) input[i] = 1;

    // do inference
    float output[o];
    for (int i = 0; i < 100; i++) {
        auto start = chrono::system_clock::now();
        alexnet.doInference(input, output);
        auto end = chrono::system_clock::now();
        cout << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << endl;
    }

    // print output
    std::cout << "\nOutput:\n";
    for (unsigned int i = 0; i < o; i++)
    {
        cout << output[i] << ", ";
        if (i % 10 == 0) cout << endl;
    }
    cout << endl;

    return 0;
}