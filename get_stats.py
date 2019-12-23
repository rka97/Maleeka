from os import listdir
from os.path import isfile, join
from generate_dataset import init_letters, GENERATED_DATASET_DIRECTORY

def main():
	
	with open("stats.txt", "w") as f:
		for letter, data in init_letters.items():
			f.write(letter + " :\n")
			forms = data["Forms"]
			for form in forms:
				f.write("\t" + form + " : ")
				images = listdir(GENERATED_DATASET_DIRECTORY + letter + '/' + form + '/')
				f.write(str(len(images)) + "\n")				

main()
