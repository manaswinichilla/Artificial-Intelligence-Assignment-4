#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: admysore- hdeshpa- machilla
#
import math
import operator

from PIL import Image, ImageDraw, ImageFont
import sys

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    # print(im.size)
    # print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [["".join(['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH)]) for y in
                    range(0, CHARACTER_HEIGHT)], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}


#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)


## Below is just some sample code to show you how the functions above work.
# You can delete this and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print("\n".join([ r for r in test_letters[2] ]))

# function to read the train_txt_fname file
def reading_the_data():
    data_list = []
    file = open(train_txt_fname, 'r')
    #read all the lines
    fline= file.readlines()
    file.close()
    for word in fline:
        #parsing
        word=word.split()
        data_list += [[word]]
    return data_list

#comparing each pixel of the image
#if space-' ', increasing the count of spaces by 1
# if the test letters is the same as the train letter, increasing the character_match by 1
# else, character_mismatch++
def pixel_comparison(flag):
    if flag==0:
        spaces_encountered=1
        return spaces_encountered
    if flag==1:
        character_match =1
        return character_match
    else:
        character_mismatch =1
        return character_mismatch

#probability for the spaces encountered
def space_probability(spaces_encountered):
    # 14*25=350
    return spaces_encountered/float(350)

#probability for */ black spaces
def char_probability(character_match,character_mismatch ):
    return character_match/float(character_mismatch)

#for each character determined, we calculate the probability
def character_probability(character_match,character_mismatch, spaces_encountered):
    # 340 because total pixels=350
    # we tried it for other values, some of it gives an excellent output for a few images while poor results for the other images
    if spaces_encountered > 340:
            #14*25=350
        letter_determined = space_probability(spaces_encountered)
    else:
        letter_determined = char_probability(character_match, character_mismatch)
    return letter_determined

# checking to see what each pixel has;
#function to determine the character by comparing train and test
def simple_probability_calculation(test_letters):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_determined={}
    for i in TRAIN_LETTERS:
        character_match = 0
        spaces_encountered = 0
        character_mismatch = 1
        for idx in range(0, len(test_letters)):
            for img_pixel in range(0, len(test_letters[idx])):
                if train_letters[i][idx][img_pixel] == ' ' and test_letters[idx][img_pixel] == ' ':
                    spaces_encountered +=pixel_comparison(0)
                else:
                    if train_letters[i][idx][img_pixel] == test_letters[idx][img_pixel]:
                        character_match +=pixel_comparison(1)
                    else:
                        character_mismatch +=pixel_comparison(2)
        letter_determined[' '] = 0.4
        letter_determined[i]=character_probability(character_match, character_mismatch, spaces_encountered)
    return letter_determined

#for each entry in the transition state dictionary, we calculate the frequency
def transition_word_count():
    data_list = reading_the_data()
    transition_state_dictionary = {}
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    for each_string in data_list:
        str = (" ").join(each_string[0])
        for char in range(0, len(str) - 1):
            #if already in the transition state dictionary, incrementing by 1
            if (str[char] in TRAIN_LETTERS) and (str[char + 1] in TRAIN_LETTERS) and transition_state_dictionary.__contains__(str[char] + "$" + str[char + 1]):
                transition_state_dictionary[str[char] + "$" + str[char + 1]] = transition_state_dictionary[str[char] + "$" + str[char + 1]] + 1
            else:
                #else, setting the value to 1
                if (str[char] in TRAIN_LETTERS) and (str[char + 1] in TRAIN_LETTERS):
                    transition_state_dictionary[str[char] + "$" + str[char + 1]] = 1
    return transition_state_dictionary, data_list

#function to calculate the total sum of every key entry in the transition state dictionary
def transition_sum(TRAIN_LETTERS,transition_state_dictionary ):
    transitions_sum_dictionary = {}
    for char in range(0, len(TRAIN_LETTERS)):
        probabilitysum = 0
        for every_key_pair in transition_state_dictionary.keys():
            if (TRAIN_LETTERS[char] == every_key_pair.split('$')[0]):
                probabilitysum += transition_state_dictionary[every_key_pair]
        if probabilitysum != 0:
            transitions_sum_dictionary[TRAIN_LETTERS[char]] = probabilitysum
    return transitions_sum_dictionary

#function to update the key value of every character
def total_probability_transition_probability(transition_state_dictionary):
    totalprobability = sum(transition_state_dictionary.values())
    for every_key_pair in transition_state_dictionary.keys():
        transition_state_dictionary[every_key_pair] = transition_state_dictionary[every_key_pair] / float(totalprobability)
    return transition_state_dictionary

#function to get transition probability for all possible letters in training
def transition_probability():
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    transition_state_dictionary, data_list=transition_word_count()
    transitions_sum_dictionary= transition_sum(TRAIN_LETTERS, transition_state_dictionary)
    for every_key_pair in transition_state_dictionary.keys():
        transition_state_dictionary[every_key_pair] = (transition_state_dictionary[every_key_pair]) / (float(transitions_sum_dictionary[every_key_pair.split("$")[0]]))
    transition_state_dictionary=total_probability_transition_probability(transition_state_dictionary)
    return data_list,transition_state_dictionary
    # print(trasition_state_dictionary)

#function to get frequency of the first character
def first_char(TRAIN_LETTERS):
    each_img, transition_state_dictionary=transition_probability()
    frequencies_of_first_character = {}
    for each_string in each_img:
        for first_string_word in each_string[0]:
            if first_string_word[0] in TRAIN_LETTERS:
                # if already in dictionary, increment by 1
                if frequencies_of_first_character.__contains__(first_string_word[0]):
                    frequencies_of_first_character[first_string_word[0]] += 1
                else:
                    # else, setting the value to 1
                    frequencies_of_first_character[first_string_word[0]] = 1
    return frequencies_of_first_character, each_img, transition_state_dictionary

#function to calculate the frequency of every character
def each_character(TRAIN_LETTERS, each_img):
    numberofcharacters = 0
    # frequency of each character
    each_character_frequency = {}
    for every_string in each_img:
        str = (" ").join(every_string[0])
        for char in str:
            if char in TRAIN_LETTERS:
                numberofcharacters = numberofcharacters + 1
                if each_character_frequency.__contains__(char):
                    each_character_frequency[char] += 1
                else:
                    each_character_frequency[char] = 1
    return  each_character_frequency, numberofcharacters

#function to calculate the likelihood of each state, given it is in a certain hidden state
def emission_probability():
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    frequencies_of_first_character, each_img, transition_state_dictionary=first_char(TRAIN_LETTERS)
    totalprobability = sum(frequencies_of_first_character.values())
    for each_key_pair in frequencies_of_first_character.keys():
        frequencies_of_first_character[each_key_pair] = frequencies_of_first_character[each_key_pair] / float(totalprobability)
    each_character_frequency, numberofcharacters =each_character(TRAIN_LETTERS, each_img)
    for each_key_pair in each_character_frequency.keys():
        each_character_frequency[each_key_pair] = (each_character_frequency[each_key_pair] ) / (float(numberofcharacters) + math.pow(10, 10))
    # print(each_character_frequency)
    totalprobability = sum(each_character_frequency.values())
    for each_key_pair in each_character_frequency.keys():
        each_character_frequency[each_key_pair] = each_character_frequency[each_key_pair] / float(totalprobability)
    # print(each_character_frequency)
    return frequencies_of_first_character,each_character_frequency, transition_state_dictionary

#for each test_letter, we calculate the frequencies and take the letter which has the highest probability
def simple_bayes_net(test_letters):
    output_word = ''
    for each_letter in test_letters:
        letter = simple_probability_calculation(each_letter)
        max_probability_letter = max(letter.items(), key=operator.itemgetter(1))[0]
        output_word += max_probability_letter
        #print(output_word)
    return output_word

#for each letter determined, we calculate its probability
def highest_probable_letter(test_letters):
    letter_determined = simple_probability_calculation(test_letters)
    # print(letter_determined)
    probability_total = 0
    for each_key_pair in letter_determined.keys():
        if each_key_pair != " ":
            probability_total = probability_total + letter_determined[each_key_pair]
        else:
            probability_total = probability_total + 1
    for each_key_pair in letter_determined.keys():
        if each_key_pair != " ":
            if letter_determined[each_key_pair] != 0:
                letter_determined[each_key_pair] = letter_determined[each_key_pair] / float(probability_total)
            else:
                letter_determined[each_key_pair] = 0.002
    return letter_determined

#we take the 4 most probable letters
def most_probable_letters(test_letters):
    letter_determined=highest_probable_letter(test_letters)
    top_four_probable_letters = dict(sorted(letter_determined.items(), key=operator.itemgetter(1), reverse=True)[:4])
    return top_four_probable_letters

#the most probable letter determined for the given the test letter
def initial_probability(test_letters):
    initial_probability = most_probable_letters(test_letters[0])
    return initial_probability

#Using the initial, transition and emission probabilities, we calculate the probability of each letter and store it.
# For each letter prediction, we pick the letter which has the highest probability.
#taking log proababilitie to handle underflow.
def hmm_probability(TRAIN_LETTERS,init_probability, vMat_map, listofchars):
    for row in range(0,len(TRAIN_LETTERS)):
        if TRAIN_LETTERS[row] in init_probability and init_probability[TRAIN_LETTERS[row]]!=0 and TRAIN_LETTERS[row] in frequencies_of_first_character:
            vMat_map[row][0] = [- math.log10(init_probability[TRAIN_LETTERS[row]]),'p']
    for column in range(1,len(test_letters)):
        letter_determined = most_probable_letters(test_letters[column])
        # print(letter_determined)
        if ' ' in letter_determined:
            listofchars[column] = " "
        for each_key_pair in letter_determined.keys():
            temporary = {}
            for row in range(0,len(TRAIN_LETTERS)):
                if each_key_pair in letter_determined and (TRAIN_LETTERS[row]+"$"+each_key_pair) in trasition_state_dictionary:
                    temporary[TRAIN_LETTERS[row]] = 0.0002 * vMat_map[row][column-1][0]- math.log10(trasition_state_dictionary[TRAIN_LETTERS[row]+"$"+each_key_pair])- 10 * math.log10(letter_determined[each_key_pair])
            max = 0
            Maxkey = ''
            for tempkey in temporary.keys():
                if max < temporary[tempkey]:
                    max = temporary[tempkey]
                    Maxkey = tempkey
            if Maxkey != '':
                vMat_map[TRAIN_LETTERS.index(each_key_pair)][column] = [temporary[Maxkey],Maxkey]
    return vMat_map, listofchars

#function to store the viterbi map values of all characters.
def hmm(test_letters):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    listofchars = ['']*len(test_letters)
    vMat_map = []
    for each_char in range(0, len(TRAIN_LETTERS)):
        temporary = []
        for each_char1 in range(0, len(test_letters)):
            temporary.append([0,''])
        vMat_map.append(temporary)
    init_probability = initial_probability(test_letters)
    vMat_map, listofchars= hmm_probability(TRAIN_LETTERS, init_probability, vMat_map, listofchars)
    return vMat_map,listofchars

#function to keep track of max value of character_determined
def map_hmm_calculation_first_char(TRAIN_LETTERS,vMat_map, listofchars):
    maximum = math.pow(10, 10)
    for row in range(0, len(TRAIN_LETTERS)):
        if vMat_map[row][0][0] != 0 and vMat_map[row][0][0] < maximum :
            maximum = vMat_map[row][0][0]
            listofchars[0] = TRAIN_LETTERS[row]
    return listofchars

#calculating using previous column value and keeping track of max value
#appending the results to a string
def map_hmm_calculation(test_letters):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    vMat_map,listofchars =hmm(test_letters)
    finalresult=""
    listofchars=map_hmm_calculation_first_char(TRAIN_LETTERS, vMat_map, listofchars)
    for column in range(1, len(test_letters)):
        minimum = math.pow(10, 10)
        for row in range(0, len(TRAIN_LETTERS)):
            if vMat_map[row][column][0] != 0 and minimum > vMat_map[row][column][0] and listofchars[column]!=' ' and row != len(TRAIN_LETTERS)-1:
                minimum = vMat_map[row][column][0]
                listofchars[column] = TRAIN_LETTERS[row]
    return finalresult.join(listofchars)

# The final two lines of your output should look something like this:
frequencies_of_first_character, each_character_frequency, trasition_state_dictionary = emission_probability()
print("Simple: " + simple_bayes_net(test_letters))
print("   HMM: " + map_hmm_calculation(test_letters))



