from nltk.corpus import wordnet as wn
poses = {
    'a': 'adj', 'n': 'noun', 'v': 'verb', 's': 'adj(s)', 'r': 'adv'

}
word = input("Please enter a word: ")
breakpoint()
for synset in wn.synsets(word):
    print("{}: {}".format(poses[synset.pos()], ", ".join([l.name() for l in synset.lemmas()])))