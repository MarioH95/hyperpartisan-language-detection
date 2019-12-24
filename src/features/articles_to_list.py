import xml.etree.ElementTree as ET
import pickle
from tqdm import tqdm
import mmap

#number of lines for tqdm progress bar
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
#retrieve xml file with article id, hyperpartisan tag etc.    
tree = ET.parse('C:\\Users\\MarioH\\Desktop\\Uni\\Dissertation\\labels\\test_by-publisher\\ground-truth-training-bypublisher-20181122.xml')
root = tree.getroot()

tags = []
article_id = []
#get the hyperpartisan tag and id and add them to their seperate lists
for child in root:
	tags.append(child.attrib['hyperpartisan'])
	article_id.append(int(child.attrib['id']))

articles = []
#get articles bodies from text file and separately add them to list
with open('C:\\Users\\MarioH\\Desktop\\Uni\\Dissertation\\articles\\test_by-publisher\\articles-training-bypublisher-20181122.xml_clean.txt', encoding="utf-8") as input_file:
	prevNum = article_id[0]
	currentArticle = ""
	count = 0

	for line in tqdm(input_file, total=get_num_lines('C:\\Users\\MarioH\\Desktop\\Uni\\Dissertation\\articles\\test_by-publisher\\articles-training-bypublisher-20181122.xml_clean.txt')):
		intNumber = int(line.split()[0])
		if prevNum != intNumber:
			articles.append(currentArticle)
			currentArticle = ""
			prevNum = intNumber
			count = count + 1
		currentArticle = currentArticle + " " + line.split("\t", 1)[1].strip('\n')

if len(articles) < len(article_id):
	articles.append(currentArticle)

#pickle the tags and seperated articles
pickle.dump(tags, open("traintags.pickle", "wb"))
pickle.dump(articles, open("trainarticles.pickle", "wb"))