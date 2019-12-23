from skimage import io
import pathlib
from common import *

init_letters = {
	"ا": {
		"Forms": ["Isolated", "Final"],
		"Letter": "1575",
	},
	"ب": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1576"
	},
	"ت": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1578"
	},
	"ث": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1579"
	},
	"ج": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1580"
	},
	"ح": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1581"
	},
	"خ": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1582"
	},
	"د": {
		"Forms": ["Isolated", "Final"],
		"Letter": "1583"
	},
	"ذ": {
		"Forms": ["Isolated", "Final"],
		"Letter": "1584"
	},
	"ر": {
		"Forms": ["Isolated", "Final"],
		"Letter": "1585"
	},
	"ز": {
		"Forms": ["Isolated", "Final"],
		"Letter": "1586"
	},
	"س": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1587"
	},
	"ش": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1588"
	},
	"ص": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1589"
	},
	"ض": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1590"
	},
	"ط": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1591"
	},
	"ظ": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1592"
	},
	"ع": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1593"
	},
	"غ": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1594"
	},
	"ف": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1601"
	},
	"ق": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1602"
	},
	"ك": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1603"
	},
	"ل": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1604"
	},
	"م": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1605"
	},
	"ن": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1606"
	},
	"ه": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1607"
	},
	"و": {
		"Forms": ["Isolated", "Final"],
		"Letter": "1608"
	},
	"ي": {
		"Forms": ["Isolated", "Initial", "Medial", "Final"],
		"Letter": "1610"
	},
	"لا": {
		"Forms": ["Isolated", "Final"],
		"Letter": "65275"
	}
}

left_disconnected_letters = ["ا", "د", "ذ", "ر", "ز", "و"]


GENERATED_DATASET_DIRECTORY = "generated_dataset/"


def generate_folders():
	for letter, data in init_letters.items():
		for form in data["Forms"]:
			pathlib.Path(GENERATED_DATASET_DIRECTORY + letter +
						 "/" + form).mkdir(parents=True, exist_ok=True)

# returns Parts of Arabic Word


def get_PAWs(word):
	PAWs = []
	last_idx = 0
	for idx, letter in enumerate(word):
		if letter in left_disconnected_letters:
			PAWs.append(word[last_idx:idx+1])
			last_idx = idx + 1
	if last_idx < len(word):
		PAWs.append(word[last_idx:])
	return PAWs

def save_letter(segmented_letter, txt_letter, form, word_idx, idx):
	img_name = str(word_idx) + "_" + str(idx) + '.png'
	directory_name = GENERATED_DATASET_DIRECTORY + txt_letter + "/" + form
	io.imsave(directory_name + "/" + img_name, segmented_letter * 255)
	print("saved" , "to" , directory_name, img_name)
	show_images([segmented_letter])



def generate_dataset(words_characters, filename):
	txt_file = open(filename, encoding='utf-8')
	txt_words = txt_file.read().split(' ')
	generate_folders()
	for word_idx, (segmented_word, txt_word) in enumerate(zip(words_characters, txt_words)):
		if len(segmented_word) != len(txt_word):
			continue
		PAWs = get_PAWs(txt_word)
		idx = 0
		print("PAWs", PAWs)
		for PAW in PAWs:
			if len(PAW) == 1:
				# isolated letter
				save_letter(segmented_word[idx], PAW[0], "Isolated", word_idx, idx)
				idx += 1
			else :
				for paw_letter_idx, paw_letter in enumerate(PAW):
					if paw_letter_idx == 0:
						# Initial letter
						save_letter(segmented_word[idx], paw_letter, "Initial", word_idx, idx)
					elif paw_letter_idx == len(PAW) - 1:
						# Final letter
						save_letter(segmented_word[idx], paw_letter, "Final", word_idx, idx)
					else:
						# Medial letter
						save_letter(segmented_word[idx], paw_letter, "Medial", word_idx, idx)
					idx += 1
def test():
	words_string = "ابراهيم محمود احمد عصام سار ماهر احمد خاالد"
	words_characters = []
	words = words_string.split(" ")
	for word in words:
		words_characters.append(list(word))
	generate_dataset(words_characters, "ss")

# test()