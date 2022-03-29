# Set - 1

## Experiment 1 -MITBIH
python eval.py --model CnnWithResidualConnection --data mitbih --count-params

Scores on test Data:: Accuracy: 0.9866617942627444, F1: 0.9229576310626472




## Experiment 2 a [PTB]


python eval.py --model CnnWithResidualConnectionPTB --data ptbdb --count-params
Scores on test Data:: Accuracy: 0.9958777052559258, F1: 0.9948527374854859
auc-roc:  0.9976518919529226
Logistic: f1=0.995 auc_prc=0.999

# Set - 1 (New) Shubham

## Experiment 0 a
python eval.py --model VanillaCnnMITBIH --data mitbih --count-params
Scores on test Data:: Accuracy: 0.9808605883427736, F1: 0.8985743451164183

## Experiment 0 b
python eval.py --model VanillaCnnPTB --data ptbdb --count-params
Scores on test Data:: Accuracy: 0.9769838543455857, F1: 0.9709807177413295
auc-roc:  0.9928951060794416
Logistic: f1=0.971 auc_prc=0.996



## Experiment 1 a

CNN Resnet 100 epochs
- Validation:
  - Accuracy: 
  - F1: 
- Test:
  - Accuracy: 
  - F1: 

## Experiment 1 d
CNN Resnet + Inverse Frequency offset by median weighting
- Validation:
  - Accuracy: 
  - F1: 
- Test:
  - Accuracy: 
  - F1: 

## Experiment 2 a [on PTB]
CNN + Residual Connection
(at the end of epoch 95)
- Validation:
  - Accuracy: 99.79
  - F1: 99.73
- Test:
  - Accuracy: 99.59
  - F1: 99.49

## Experiment 4 a [on PTB]

Bidirectional RNN Model with 21 elements fed at a time 
(at the end of epoch 71)
python eval.py --model RnnModelPTB --data ptbdb --count-params
Scores on test Data:: Accuracy: 0.9776709034695981, F1: 0.9721509411190695
auc-roc:  0.9939553712457029
Logistic: f1=0.972 auc_prc=0.997


## Experiment 5 a [on MITBIH]
Bidirectional RNN Model with 21 elements fed at a time 
(at end of epoch 97)
 python eval.py --model RnnModelMITBIH --data mitbih --count-params

Scores on test Data:: Accuracy: 0.9823223095194592, F1: 0.8996577428600924





## Experiment 6a
Total num. of trainable parameters in the model: 74037
Total num. of  parameters in the model: 74037
Scores on test Data:: Accuracy: 0.9785766490042025, F1: 0.8899904967548812
Scores on validation data:: Accuracy: 0.9935469133687397, F1: 0.9647826633367904
## Experiment 6b
python eval.py --model CnnModel2DPTB --data ptbdb --count-params
Total num. of trainable parameters in the model: 73905
Total num. of  parameters in the model: 73905
Scores on test Data:: Accuracy: 0.9817931982136723, F1: 0.9773096050656778
Logistic: f1=0.977 auc=0.998
Scores on validation data:: Accuracy: 0.9832546157148991, F1: 0.9791436906512732
Logistic: f1=0.979 auc=0.998

## Experiment 10 a
python eval.py --model VanillaRNNPTB --data ptbdb --count-params
Scores on test Data:: Accuracy: 0.9766403297835795, F1: 0.9706741618310768
auc-roc:  0.9932679336531575
Logistic: f1=0.971 auc_prc=0.997

## Experiment 10 b
python eval.py --model VanillaRNNMITBIH --data mitbih --count-params
Scores on test Data:: Accuracy: 0.9791247944454595, F1: 0.8901759009266896


# Set - 2

# Set - 3

