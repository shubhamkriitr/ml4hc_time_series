# Set - 1

## Experiment 1 -MITBIH

At the end of 50th epoch
  Val F1: 92.04
 Val acc: 98.5
 Test F1: 90.07
Test acc: 98.35

## Experiment 2 a [PTB]
Best Values:
Val F1: 99.46
Val Acc: 99.57
test F1: 99.23
test acc: 99.38

# Set - 1 (New) Shubham

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

RNN Model with 21 elements fed at a time 
(at the end of epoch 71)
- Validation:
  - Accuracy: 98.37
  - F1: 91.98
- Test:
  - Accuracy: 98.23
  - F1: 89.97
## Experiment 5 a [on MITBIH]
RNN Model with 21 elements fed at a time 
(at end of epoch 97)

- Validation:
  - Accuracy: 97.94
  - F1: 97.43
- Test:
  - Accuracy: 9.77
  - F1: 97.22


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

## Experiment 7 a [on MITBIH]
Transformers

# Set - 2

# Set - 3

