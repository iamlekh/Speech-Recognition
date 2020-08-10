import librosa
import os
import json

dataset_path = '/home/darpan/Documents/speech rec/speech_commands'
json_path = "data.json"
samples_to_consider = 22050

def prepare_data(dataset_path, json_path, n_mfcc= 13, hop_length = 512, n_fft = 2048):
  data = {
      "mapping":[],
      'labels': [],
      "MFCCs":[],
      'files':[]
  }

  for i,(dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
      
    if dirpath not in dataset_path:
          label = dirpath.split("/")[-1]
          data["mapping"].append(label)
          print("\nProcessing: '{}'".format(label))

          # process all audio files in sub-dir and store MFCCs
          for f in filenames:
              file_path = os.path.join(dirpath, f)

              # load audio file and slice it to ensure length consistency among different files
              signal, sample_rate = librosa.load(file_path)

              # drop audio files with less than pre-decided number of samples
              if len(signal) >= samples_to_consider:
                    
                  # ensure consistency of the length of the signal
                  signal = signal[:samples_to_consider]

                  # extract MFCCs
                  MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)

                  # store data for analysed track
                  data["MFCCs"].append(MFCCs.T.tolist())
                  data["labels"].append(i)
                  data["files"].append(file_path)
                  print("{}: {}".format(file_path, i))

    with open(json_path, "w") as fp:
      json.dump(data, fp, indent=4)



# prepare_data(dataset_path,json_path,13 ,512, 2048)

if __name__ == "__main__":
    prepare_data(dataset_path,json_path)