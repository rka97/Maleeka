#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <sstream>
using std::stringstream;

int main() {
    char* outText;

    tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
    if (api->Init(NULL, "ara")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

    Pix* image = pixRead("output.png");
    api->SetImage(image);

    Boxa* boxes = api->GetComponentImages(tesseract::RIL_SYMBOL, TRUE, nullptr, nullptr);
    printf("Found %d textline image components.\n", boxes->n);
    Pix* thresholded_image = api->GetThresholdedImage();
    // pixaWrite("words.png", cropped_pics);
    boxaWrite("boxes.txt", boxes);
    for (int i = 0; i < boxes->n; i++) {
        BOX* box = boxaGetBox(boxes, i, L_CLONE);
        auto cropped_pic= pixClipRectangle(thresholded_image, box, nullptr);
        stringstream word;
        api->SetRectangle(box->x, box->y, box->w, box->h);
        char* ocrResult = api->GetUTF8Text();
        int conf = api->MeanTextConf();
        if (conf > 40) {
            if(strlen(ocrResult) == 2) {
                word << "output/" << ocrResult << "-" << i << ".png";
                pixWrite(word.str().c_str(), cropped_pic, IFF_PNG);
            }
        }
        fprintf(stdout, "Box[%d]: x=%d, y=%d, w=%d, h=%d, confidence: %d, text: %s", 
                i, box->x, box->y, box->w, box->h, conf, ocrResult);
        // break;
    }
    
    // outText = api->GetUTF8Text();
    // printf("OCR output:\n%s", outText);
    // delete[] outText;
    api->End();
    pixDestroy(&image);
    // pixDestroy(&thresholded_image);
    return 0;
}