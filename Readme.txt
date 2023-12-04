# CNN-SWS-f


CNN-based Auto-picking method for determining end time window of local shear-wave splitting measurements

Versions:
        python 3.6.9
        numpy 1.19.5
        Tensorflow 1.14.0
        Keras 2.2.5

**Please report confusions/errors to yzcd4@umsystem.edu or sgao@mst.edu

The technique is decribed in the paper: 

------------------------------------------------

Data structure:

Data, in the paper (Zhang and Gao, 2024), should be download from figshare (https://figshare.com/articles/dataset/CNN_Local_SWS_f/24714855) and unzip under ~/data.

All sac files must be stored at:
~/data/SWS_trace/stname_NW/EQ12345678901/stname_NW.n

stname_NW:
        stname is station name. If station name is less than 6 character, shoud be filled by 'x'.
        NW is name of network.

EQ12345678901:
        Event name
        12 is for year.
        345 is for Julday.
        67 is for hour.
        89 is for minute.
        01 is for second.

stname_NW.n:
        stname_NW is same with previous one.
        n represents component.
        Each event should have 3 components (z, n, e).

The list of event must be at ~/data/train.list
        train.list is for training, and test.list is for testing.
        The list is contain 13 columns (* represents that value is not important of this project):
                1st is stname_NW.
                2nd is EQ12345678901.
                3rd is station latitude.
                4th is station longitude.
                *5th is fast direction.
                *6th is standard deviation of fast direction.
                *7th is splitting time.
                *8th is standard deviation of splitting time.
                *9th is back azimuth.
                10th is event latitude.
                11th is event longitude.
                12th is event depth.
                *13th is ranking of measurement.

--------------------------------------------------

Training process:

Run Do_train.cmd to train CNN.
The parameters can be changed at ~/train/2_train/parameter.list
Link of training dataset in the paper (Zhang and Gao, 2024): https://figshare.com/articles/dataset/CNN_Local_SWS_f/24714855
        Download under ~/data/ 
        Unzip to use it.

---------------------------------------------------


Testing process:

Run Do_load.cmd to run CNN.
The parameters can be changed at ~/test/parameter.list
CNN model of paper is at ~/model/model_paper.h5
Output is at ~/test/Outp/stname_NW_EQ12345678901_000000.xxxx
        000000 is number of measurement.
        xxxx: .in is input of CNN (3 components seismogram).
              .out is target of CNN (human labeled f with a mask of gaussian distribution).
              .info is infomation of measurements (sac headers (b, o, f), theorical S arrival time, and shifted points).
              .res is output of CNN.
Link of testing dataset in the paper (Zhang and Gao, 2024): https://figshare.com/articles/dataset/CNN_Local_SWS_f/24714855
        Download under ~/data/
        Unzip to use it.

