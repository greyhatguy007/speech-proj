# Speech Recognition Application with PyQt

This Python application allows you to convert audio files, particularly MP3 files, to WAV format and then transcribe the WAV files to text using Google's Speech Recognition service. It provides a user-friendly graphical interface built with PyQt5.

## Features

- Import audio files in MP3 format.
- Convert MP3 files to WAV format.
- Perform automatic speech recognition on the audio content.
- Display the recognized text in the application's user interface.


## Speech Recognition Model

This speech recognition application leverages Google's advanced speech recognition model, which combines the power of Hidden Markov Models (HMMs) and Deep Neural Networks (DNNs) for which the backbone is **Long Short Term Memory (LSTM)** networks.

### Hidden Markov Models (HMMs)

Hidden Markov Models are a statistical model used in various speech recognition systems. They are particularly effective in modeling the acoustic properties of speech, making them an essential component in many automatic speech recognition (ASR) systems.

<center>
 
![HMM](/public/HMM.png)  
 *overview of HMM*  
 </center>  

### Deep Neural Networks (DNNs)

Deep Neural Networks have revolutionized the field of speech recognition in recent years. They are known for their ability to capture complex patterns and dependencies in audio data. DNNs are used to refine the results obtained from HMMs, resulting in more accurate transcription of spoken language.

This hybrid approach, which combines the strengths of HMMs and DNNs, allows the model to deliver high-quality speech recognition and transcription.

For more information on the technology behind this model, please refer to Google's official documentation and research papers.

<center>
 
![HMM](/public/DNN.png)  
 *overview of DNN*  
 </center>  
 
### Long Short-Term Memory (LSTM) Networks

LSTM networks are a type of recurrent neural network (RNN) that excel in capturing long-range dependencies in sequential data, such as speech. The inclusion of LSTM networks in the model enhances its ability to understand and transcribe spoken language.

This hybrid approach, combining HMMs, DNNs, and LSTMs, allows the model to deliver high-quality speech recognition and transcription, making it suitable for a wide range of applications.


A simple code to generate a LSTM model using pytorch is as follows  

```python
import torch
import torch.nn as nn

class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SpeechRecognitionModel, self).__init()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        # out: (batch_size, sequence_length, hidden_size*2)
        out = self.fc(out[:, -1, :])  # Last time step output
        return out


#hyperparameters
input_size = 13  # Input features (MFCC coefficients)
hidden_size = 256
num_layers = 3
num_classes = 29  # Number of classes (phonemes)

# Model Instance
model = SpeechRecognitionModel(input_size, hidden_size, num_layers, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
**Recurrent Neural Networks (RNNs)** serve as the foundation for **Long Short-Term Memory (LSTM) networks**. LSTMs are a specialized type of RNN designed to address the vanishing gradient problem often encountered in standard RNNs. They are distinguished by their ability to capture long-range dependencies in sequential data, thanks to memory cells and gating mechanisms. These components control the flow of information, allowing LSTMs to retain and update information over extended sequences. LSTMs are particularly effective for tasks like speech recognition, natural language processing, and time series analysis, where preserving long-term context is crucial for accurate predictions and understanding temporal relationships.

<center>
 
![HMM](/public/RNN.png)  
 *overview of RNN*  
 </center> 

## Prerequisites For the Application

- Python 3.x
- PyQt5
- PyDub
- SpeechRecognition

Install the required packages using the following command:

To run the code
*optional steps - create a virtual environment*

```bash
git clone

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python main.py
```

## Usage

1. Run the application by executing main.py.
2. Click the "Import Audio File" button to select an MP3 audio file for conversion and transcription.
3. The application will convert the audio to WAV format and transcribe the content using Google's Speech Recognition service.
4. The recognized text will be displayed in the result area.

## Application Interface

<center>
 
![Interface 1](/public/SS1.png)
 *Initiating venv and running the Program* 

 </center> 

<center>
 
![Interface 2](/public/SS2.png)
 *Model Anayzing Imported audio file* 

 </center> 

<center>
 
![Interface 3](/public/SS3.png)
 *Result Displayed* 

 </center> 
 

### [Application Functioning](./public/demo.mkv)