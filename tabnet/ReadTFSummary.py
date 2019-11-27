import tensorflow as tf
import MakePlots as plot

path_to_events_file = "tflog/tabnet_forest_covertype_model/events.out.tfevents.1574818339.cardinalmoose"

stepArr = []
trainLoss = []
valLoss = []
valAcc = []
testAcc = []

#This is hard-coded for now...
for i in range(1,21):
    stepArr.append(i)

for e in tf.train.summary_iterator(path_to_events_file):
    for v in e.summary.value:
        if(v.tag == 'Total_loss'):
            valLoss.append(v.simple_value)
        if(v.tag == 'Val_accuracy'):
            valAcc.append(v.simple_value)
        if(v.tag == 'Test_accuracy'):
            testAcc.append(v.simple_value)

print(len(stepArr))
print(len(valLoss))
print(len(valAcc))
plot.PlotLoss(stepArr,valLoss,valLoss)
plot.PlotAcc(stepArr,valAcc,testAcc)


#import numpy as np
#from tensorflow.python.summary.event_accumulator import EventAccumulator

#import matplotlib as mpl
#import matplotlib.pyplot as plt

#def plot_tensorflow_log(path):

    # Loading too much data is slow...
#    tf_size_guidance = {
#        'compressedHistograms': 10,
#        'images': 0,
#        'scalars': 100,
#        'histograms': 1
#    }

#    event_acc = EventAccumulator(path, tf_size_guidance)
#    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())

#    training_accuracies =   event_acc.Scalars('Test_accuracy')
#    validation_accuracies = event_acc.Scalars('Val_accuracy')

#    steps = 10
#    x = np.arange(steps)
#    y = np.zeros([steps, 2])

#    for i in xrange(steps):
#        y[i, 0] = training_accuracies[i][2] # value
#        y[i, 1] = validation_accuracies[i][2]

#    plt.plot(x, y[:,0], label='training accuracy')
#    plt.plot(x, y[:,1], label='validation accuracy')

#    plt.xlabel("Steps")
#    plt.ylabel("Accuracy")
#    plt.title("Training Progress")
#    plt.legend(loc='upper right', frameon=True)
#    plt.show()

#plot_tensorflow_log(path_to_events_file)
