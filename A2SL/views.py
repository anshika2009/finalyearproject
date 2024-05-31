from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login,logout
from django.core.signals import request_finished
from django.dispatch import receiver
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import nltk
nltk.download('punkt') #added
from django.contrib.staticfiles import finders
from django.contrib.auth.decorators import login_required
from keras.models import model_from_json
import cv2
import numpy as np

def home_view(request):
	return render(request,'home.html')


def about_view(request):
	return render(request,'about.html')


# def contact_view(request):
# 	return render(request,'contact.html')

@login_required(login_url="login")

def animation_view(request):
	if request.method == 'POST':
		text = request.POST.get('sen')
		#tokenizing the sentence
		text.lower()
		if 'name' not in text and 'i am' not in text and 'I am' not in text:
			text=correct(text)
		#tokenizing the sentencecoof
		words = word_tokenize(text)     #seperates special characters like , . 
		tagged = nltk.pos_tag(words)        #Part-of-Speech Tagging [('John', 'NNP'), ("'s", 'POS'), ('big', 'JJ')]
		tense = {}
		tense["future"] = len([word for word in tagged if word[1] == "MD"])
		tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
		tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]])
		tense["present_continuous"] = len([word for word in tagged if word[1] in ["VBG"]])


		#stopwords that will be removed
		stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've",'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then', 'the','am', 'mustn', 'nor', 'as', "it's", "needn't", 'd', 'have',  'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])

#alternative to lemmatizing
#stemming applies simpler rules to chop off prefixes or suffixes, often resulting in non-dictionary words.
		#removing stopwords and applying lemmatizing nlp process to words
		lr = WordNetLemmatizer()
		filtered_text = []
		for w,p in zip(words,tagged):
			if w not in stop_words:
				if p[1]=='VBG' or p[1]=='VBD' or p[1]=='VBZ' or p[1]=='VBN' or p[1]=='NN':
					filtered_text.append(lr.lemmatize(w,pos='v'))  #lemmatized as a verb (pos='v').
				elif p[1]=='JJ' or p[1]=='JJR' or p[1]=='JJS'or p[1]=='RBR' or p[1]=='RBS':
					filtered_text.append(lr.lemmatize(w,pos='a'))  #lemmatized as an adjective (pos='a').

				else:
					filtered_text.append(lr.lemmatize(w))


		#adding the specific word to specify tense
		words = filtered_text
		temp=[]
		for w in words:
			if w=='I':
				temp.append('Me')
			else:
				temp.append(w)
		words = temp
		probable_tense = max(tense,key=tense.get)

		if probable_tense == "past" and tense["past"]>=1:
			temp = ["Before"]
			temp = temp + words
			words = temp
		elif probable_tense == "future" and tense["future"]>=1:
			if "Will" not in words:
					temp = ["Will"]
					temp = temp + words
					words = temp
			else:
				pass
		elif probable_tense == "present":
			if tense["present_continuous"]>=1:
				temp = ["Now"]
				temp = temp + words
				words = temp


		filtered_text = []
		for w in words:
			path = w + ".mp4"
			f = finders.find(path)
			#splitting the word if its animation is not present in database
			if not f:
				for c in w:
					filtered_text.append(c)
			#otherwise animation of word
			else:
				filtered_text.append(w)
		words = filtered_text;


		return render(request,'animation.html',{'words':words,'text':text})
	else:
		return render(request,'animation.html')


def correct(sentence):
    # Create a SpellChecker object
    spell = SpellChecker()

    # Split the sentence into words
    words = sentence.split()

    # Correct misspelled words
    corrected_words = [spell.correction(word) for word in words]

    # Join the corrected words back into a sentence
    corrected_sentence = ' '.join(corrected_words)
    return corrected_sentence

def signup_view(request):
	if request.method == 'POST':
		form = UserCreationForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request,user)
			# log the user in
			return redirect('animation')
	else:
		form = UserCreationForm()
	return render(request,'signup.html',{'form':form})



def login_view(request):
	if request.method == 'POST':
		form = AuthenticationForm(data=request.POST)
		if form.is_valid():
			#log in user
			user = form.get_user()
			login(request,user)
			if 'next' in request.POST:
				return redirect(request.POST.get('next'))
			else:
				return redirect('home')
				#return redirect('animation')
	else:
		form = AuthenticationForm()
	return render(request,'login.html',{'form':form})


def logout_view(request):
	logout(request)
	return redirect("login")
	#return render(request,'home.html') use this
	#return redirect("home")

@receiver(request_finished)
def auto_logout(sender, **kwargs):
    # Perform logout action when the request has finished
    logout(request)

@login_required(login_url="login")
def signtotext(request):
    # json_file = open("signlanguagedetectionmodel48x48.json", "r")
    # model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(model_json)
    # model.load_weights("signlanguagedetectionmodel48x48.h5")
    
    # def extract_features(image):
    #     feature = np.array(image)
    #     feature = feature.reshape(1,48,48,1)
    #     return feature/255.0
    
    # cap = cv2.VideoCapture(0)
    # label = ['A', 'B', 'C', 'D', 'E', 'F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    # while True:
    #     _,frame = cap.read()
    #     cv2.rectangle(frame,(0,40),(300,300),(0, 165, 255),1)
    #     cropframe=frame[40:300,0:300]
    #     cropframe=cv2.cvtColor(cropframe,cv2.COLOR_BGR2GRAY)
    #     cropframe = cv2.resize(cropframe,(48,48))
    #     cropframe = extract_features(cropframe)
    #     pred = model.predict(cropframe) 
    #     prediction_label = label[pred.argmax()]
    #     cv2.rectangle(frame, (0,0), (300, 40), (0, 165, 255), -1)
    #     if prediction_label == 'blank':
    #         cv2.putText(frame, " ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
    #     else:
    #         accu = "{:.2f}".format(np.max(pred)*100)
    #     cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
    #     cv2.imshow("output",frame)
    #     cv2.waitKey(27)
    # return render(request,'signtotext.html')
    # cap.release()
    # cv2.destroyAllWindows()
    return render(request,'signtotext.html')

def detection(request):
    json_file = open("signlanguagedetectionmodel48x48.json", "r")   #model architecture is loaded from a JSON file.
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("signlanguagedetectionmodel48x48.h5")    #model weights is loaded from a JSON file.
    
    def extract_features(image):
        feature = np.array(image)
        feature = feature.reshape(1,48,48,1)  #processing one image at a time, 48x48 dimension, 1 is no of channel
        return feature/255.0
    
    cap = cv2.VideoCapture(0)
    label = ['A', 'B', 'C', 'D', 'E', 'F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    while True:
        _,frame = cap.read()
        # _ is true or false if image is returned or not
        #frame is an image array vector captured based on the default frames per second
        cv2.rectangle(frame,(0,40),(300,300),(0, 165, 255),1)
        cropframe=frame[40:300,0:300]
        cropframe=cv2.cvtColor(cropframe,cv2.COLOR_BGR2GRAY)
        cropframe = cv2.resize(cropframe,(48,48))
        cropframe = extract_features(cropframe)
        pred = model.predict(cropframe) 
        prediction_label = label[pred.argmax()]
        cv2.rectangle(frame, (0,0), (300, 40), (0, 165, 255), -1)
        if prediction_label == 'blank':
            cv2.putText(frame, " ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
        else:
            accu = "{:.2f}".format(np.max(pred)*100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
        cv2.imshow("output",frame)
        # cv2.waitKey(27)
        key=cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        # if key==-1:
        #     cap.release()
        #     break
    # return render(request,'signtotext.html')
    cap.release()
    cv2.destroyAllWindows()
    return render(request,'signtotext.html')