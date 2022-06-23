from pydub import AudioSegment
from pydub.utils import make_chunks

myaudio = AudioSegment.from_file("D:/SPLIT/BOSUNG/10.COW_30/COW2.wav" , "wav") 
chunk_length_ms = 1000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files
for i, chunk in enumerate(chunks):
    chunk_name = "D:/SPLIT/BOSUNG/COW/COWT2_{0}.wav".format(i)
    chunk.export(chunk_name, format="wav")
print('Finish')
