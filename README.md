# kws

### Running script

```
!git pull https://github.com/qwerty-Bk/kws.git
!pip install -r kws/requirements.txt

!wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -O speech_commands_v0.01.tar.gz
!mkdir speech_commands && tar -C speech_commands -xvzf speech_commands_v0.01.tar.gz 1> log

!mv speech_commands/ kws/speech_commands/

!python kws/train.py
```
