# parse_debate.py

"""
in: url to debate transcript, list of named speakers
out: pickle for HRC and Trump with list of sentences they said
"""

from lxml import html
import requests
import pickle

# set URL and speakers
url = 'http://www.presidency.ucsb.edu/ws/index.php?pid=118971'
speakers = ['CLINTON: ', 'TRUMP: ', 'HOLT: ']

# set up containers for Clinton spoken sentences and Trump spoken sentences
clinton_lines = []
trump_lines = []
curr_speaker = speakers[2] # this debate starts with the moderator - honestly could assign this to any non Clinton or Trump speaker because those are the only ones I am tracking

# read in lines
page = requests.get(url)
tree = html.fromstring(page.content)
lines = tree.xpath('//p')

for line in lines :
    text = line.text_content()
    speaker_matches = [text.startswith(speaker) for speaker in speakers]
    if any(speaker_matches) : # update current speaker
        curr_speaker = [i for (i, j) in zip(speakers, speaker_matches) if j][0]
    if curr_speaker is speakers[0] : # store Clinton entries
        if speaker_matches[0] :
            clinton_lines.append(text[9:]) # strip 'CLINTON: '
        else :
            clinton_lines.append(text)
    elif curr_speaker is speakers[1] : # store Trump entries
        if speaker_matches[1] :
            trump_lines.append(text[7:]) # strip 'TRUMP: '
        else :
            trump_lines.append(text)
    else : # don't store entires from other speaker(s)
        continue


