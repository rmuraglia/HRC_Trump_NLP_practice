# parse_debate.py

"""
in: urls to debate transcripts, list of named speakers
out: pickles for each group with sentences they spoke
"""

from lxml import html
import requests
import pickle

# set URLs of debates
# complete up to march (inclusive)
urls = ['http://www.presidency.ucsb.edu/ws/index.php?pid=118971', 'http://www.presidency.ucsb.edu/ws/index.php?pid=116995', 'http://www.presidency.ucsb.edu/ws/index.php?pid=115148', 'http://www.presidency.ucsb.edu/ws/index.php?pid=112718', 'http://www.presidency.ucsb.edu/ws/index.php?pid=111711']

# set up speaker lists and line containers
clinton = ['CLINTON: ']
trump = ['TRUMP: ']
moderators = ['HOLT: ', 'BLITZER: ', 'BASH: ', 'LOUIS: ', 'TAPPER: ', 'HEWITT: ', 'DINAN: ', 'COOPER: ', 'LEMON: ', 'BAIER: ', 'KELLY: ', 'WALLACE: ']
dems = ['SANDERS: ']
reps = ['CRUZ: ', 'KASICH: ', 'RUBIO: ']
speakers = clinton + trump + moderators + dems + reps
s_groups = [clinton, trump, moderators, dems, reps]

clinton_lines = []
trump_lines = []
moderator_lines = []
dems_lines = []
reps_lines = []

# read in lines
def read_debate(url) :
    page = requests.get(url)
    tree = html.fromstring(page.content)
    lines = tree.xpath('//p')
    return lines

# parse lines and assign to speakers
def parse_lines(unparsed_lines) :

    # initialize convenience variables
    parsed_lines = [ [] for i in xrange(len(s_groups)) ]
    curr_speaker = None

    # iterate through lines in debate and assign to speakers
    for line in unparsed_lines :

        # get text content of line
        text = line.text_content()
        text_trim = False

        # determine who current speaker is
        speaker_matches = [text.startswith(speaker) for speaker in speakers]
        if any(speaker_matches) : # update current speaker
            curr_speaker = [i for (i,j) in zip(speakers, speaker_matches) if j][0]
            text_trim = True # new speaker, so need to trim off their name

        # add lines to appropriate speaker
        speaker_group = [curr_speaker in group for group in s_groups]
        for i in xrange(len(speaker_group)) :
            if speaker_group[i] :
                if text_trim :
                    trimmed_text = text[len(curr_speaker):]
                    parsed_lines[i].append(trimmed_text)
                else :
                    parsed_lines[i].append(text)

    return parsed_lines

for url in urls :
    debate_lines = read_debate(url)
    all_lines = parse_lines(debate_lines)
    clinton_lines.append(all_lines[0])
    trump_lines.append(all_lines[1])
    moderator_lines.append(all_lines[2])
    dems_lines.append(all_lines[3])
    reps_lines.append(all_lines[4])

# pickle output
with open('clinton.pickle', 'wb') as f :
    pickle.dump(clinton_lines, f, pickle.HIGHEST_PROTOCOL)
with open('trump.pickle', 'wb') as f :
    pickle.dump(trump_lines, f, pickle.HIGHEST_PROTOCOL)   
with open('moderator.pickle', 'wb') as f :
    pickle.dump(moderator_lines, f, pickle.HIGHEST_PROTOCOL)
with open('dems.pickle', 'wb') as f :
    pickle.dump(dems_lines, f, pickle.HIGHEST_PROTOCOL)
with open('reps.pickle', 'wb') as f :
    pickle.dump(reps_lines, f, pickle.HIGHEST_PROTOCOL) 

# read with: 
# with open('clinton.pickle', 'rb') as f : 
    # dat = pickle.load(f)