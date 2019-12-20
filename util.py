
import matplotlib.pyplot as plt


def show_image(images):
    for img in images:
        plt.imshow(img, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()


letters = {
    "1575": {
        "Forms": ["End", "Isolated"],
        "Letter": "ا"
    },
    "1576": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ب"
    },
    "1577": {
        "Forms": ["End", "Isolated"],
        "Letter": "ة"
    },
    "1578": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ت"
    },
    "1579": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ث"
    },
    "1580": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ج"
    },
    "1581": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ح"
    },
    "1582": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "خ"
    },
    "1583": {
        "Forms": ["End", "Isolated"],
        "Letter": "د"
    },
    "1584": {
        "Forms": ["End", "Isolated"],
        "Letter": "ذ"
    },
    "1585": {
        "Forms": ["End", "Isolated"],
        "Letter": "ر"
    },
    "1586": {
        "Forms": ["End", "Isolated"],
        "Letter": "ز"
    },
    "1587": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "س"
    },
    "1588": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ش"
    },
    "1589": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ص"
    },
    "1590": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ض"
    },
    "1591": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ط"
    },
    "1592": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ظ"
    },
    "1593": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ع"
    },
    "1594": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "غ"
    },
    "1601": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ف"
    },
    "1602": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ق"
    },
    "1603": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ك"
    },
    "1604": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ل"
    },
    "1605": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "م"
    },
    "1606": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ن"
    },
    "1607": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ه"
    },
    "1608": {
        "Forms": ["End", "Isolated"],
        "Letter": "و"
    },
    "1609": {
        "Forms": ["End", "Isolated"],
        "Letter": "ى"
    },
    "1610": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "ي"
    },
    "65275": {
        "Forms": ["Beginning", "End", "Isolated", "Middle"],
        "Letter": "لا"
    }
}