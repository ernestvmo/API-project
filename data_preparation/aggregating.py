'''
import wave

infiles = [ "data/data_sliced/three/rock.00000_3.wav","data/data_sliced/three/rock.00000_4.wav","data/data_sliced/three/rock.00000_5.wav","data/data_sliced/three/rock.00000_6.wav",
            "data/data_sliced/three/rock.00000_7.wav","data/data_sliced/three/rock.00000_8.wav"]
outfile = "data/data_sliced/exampwavs/examplerock.wav"

data= []
for infile in infiles:
    w = wave.open(infile, 'rb')
    data.append( [w.getparams(), w.readframes(w.getnframes())] )
    w.close()

output = wave.open(outfile, 'wb')
output.setparams(data[0][0])
output.writeframes(data[0][1])
output.writeframes(data[1][1])
output.close()
'''

from pydub import AudioSegment
sound1 = AudioSegment.from_file("data/data_sliced/three/rock.00000_1.wav", format="wav")
sound2 = AudioSegment.from_file("data/data_sliced/three/rock.00000_2.wav", format="wav")
sound3 = AudioSegment.from_file("data/data_sliced/three/rock.00000_3.wav", format="wav")
sound4 = AudioSegment.from_file("data/data_sliced/three/rock.00000_4.wav", format="wav")
sound5 = AudioSegment.from_file("data/data_sliced/three/rock.00000_5.wav", format="wav")
sound6 = AudioSegment.from_file("data/data_sliced/three/rock.00000_6.wav", format="wav")
sound7 = AudioSegment.from_file("data/data_sliced/three/rock.00000_7.wav", format="wav")
sound8 = AudioSegment.from_file("data/data_sliced/three/rock.00000_8.wav", format="wav")
sound9 = AudioSegment.from_file("data/data_sliced/three/rock.00000_9.wav", format="wav")
sound10 = AudioSegment.from_file("data/data_sliced/three/rock.00000_10.wav", format="wav")
sound11 = AudioSegment.from_file("data/data_sliced/three/rock.00001_1.wav", format="wav")
sound12 = AudioSegment.from_file("data/data_sliced/three/rock.00001_2.wav", format="wav")
sound13 = AudioSegment.from_file("data/data_sliced/three/rock.00001_3.wav", format="wav")
sound14 = AudioSegment.from_file("data/data_sliced/three/rock.00001_4.wav", format="wav")
sound15 = AudioSegment.from_file("data/data_sliced/three/rock.00001_5.wav", format="wav")
sound16 = AudioSegment.from_file("data/data_sliced/three/rock.00001_6.wav", format="wav")
sound17 = AudioSegment.from_file("data/data_sliced/three/rock.00001_7.wav", format="wav")
sound18 = AudioSegment.from_file("data/data_sliced/three/rock.00001_8.wav", format="wav")
sound19 = AudioSegment.from_file("data/data_sliced/three/rock.00001_9.wav", format="wav")
sound20 = AudioSegment.from_file("data/data_sliced/three/rock.00001_10.wav", format="wav")
sound21 = AudioSegment.from_file("data/data_sliced/three/rock.00002_1.wav", format="wav")
sound22 = AudioSegment.from_file("data/data_sliced/three/rock.00002_2.wav", format="wav")
sound23 = AudioSegment.from_file("data/data_sliced/three/rock.00002_3.wav", format="wav")
sound24 = AudioSegment.from_file("data/data_sliced/three/rock.00002_4.wav", format="wav")
sound25 = AudioSegment.from_file("data/data_sliced/three/rock.00002_5.wav", format="wav")
sound26 = AudioSegment.from_file("data/data_sliced/three/rock.00002_6.wav", format="wav")
sound27 = AudioSegment.from_file("data/data_sliced/three/rock.00002_7.wav", format="wav")
sound28 = AudioSegment.from_file("data/data_sliced/three/rock.00002_8.wav", format="wav")
sound29 = AudioSegment.from_file("data/data_sliced/three/rock.00002_9.wav", format="wav")
sound30 = AudioSegment.from_file("data/data_sliced/three/rock.00002_10.wav", format="wav")
sound31 = AudioSegment.from_file("data/data_sliced/three/rock.00003_1.wav", format="wav")
sound32 = AudioSegment.from_file("data/data_sliced/three/rock.00003_2.wav", format="wav")


# sound1 6 dB louder
louder = sound1 + 6


# sound1, with sound2 appended (use louder instead of sound1 to append the louder version)
combined = sound1 + sound2 + sound3 + sound4 + sound5 + sound6 + sound7 + sound8 + sound9 +sound10 + sound11 + sound12 + sound13 + sound14 +sound15 +sound16 +sound17 +sound18 +sound19 +sound20 +sound21 +sound22 +sound23 +sound24 +sound25 +sound26 +sound27 +sound28 +sound29 +sound30 +sound31 +sound32


# simple export
file_handle = combined.export("data/data_sliced/aggregwavs/examplerock.wav", format="wav")