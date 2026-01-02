####################
### Contributors ###
####################

Shannon M. Locke [1,3,*,**]
Elon Gaffin-Cahn [1,*]
Nadia Hosseinizaveh [1]
Pascal Mamassian [3]
Michael S. Landy [1,2]

[1] Department of Psychology, New York University, New York, NY, United States
[2] Center for Neural Science, New York University, New York, NY, United States
[3] Laboratoire des Systèmes Perceptifs, Département d’Études Cognitives, École Normale Supérieure, PSL University, CNRS, 75005 Paris, France

* Authors contributed equally to this publication
** Corresponding author. Address: 29 rue d'Ulm, 75005 Paris, France. Telephone: +33(0)144322791. Email: shannon.m.locke@nyu.edu.


################
### Citation ###
################

Locke, S.M.*, Gaffin-Cahn, E.*, Hosseinizaveh, N., Mamassian, P., & Landy, M.S. (2020). Priors and payoffs in confidence judgments. Attention, Perception, & Psychophysics. doi:10.3758/s13414-020-02018-x.


################
### Stimulus ###
################

Gabor patch tilted slightly left (Stimulus=0) or right (Stimulus=1) of vertical. Response is left (Response=0) or right (Response=1). Training/thresholding (Training=1; Session=0; Condition=0) was performed for each participant, then fixed for the main experiment (Training=0; Session>1; Condition>1). Difficulty is the absolute orientation magnitude displayed in degrees. Gabor spatial frequency was 2 cycles/deg, peak contrast 10%, Gaussian envelope (.5 deg SD), randomized phase.


########################
### Confidence scale ###
########################

Subjective confidence in probability correct, either low (Confidence=0) or high (Confidence=1). Confidence was not extrinsically rewarded (e.g., with points). No confidence judgments during Training.


####################
### Instructions ###
####################

Indicate via keypress if the stimulus was tilted left or right of vertical. Then indicate if you believe there is a low probability you are correct, or a high probability you are correct. Answer as quickly and accurately as possible, and try to earn as many points as you can. Find your strategy within 50 trials and keep it stable.


#####################
### Manipulations ###
#####################

Prior ( p(Stimulus=1) ) x Reward (reward Stimulus=1 : reward Stimulus=0)

Condition: Prior, Reward
0: .50, 0:0
1: .50, 3:3
2: .75, 3:3
3: .25, 3:3
4: .50, 4:2
5: .50, 2:4
6: .75, 2:4
7: .25, 4:2

NB: The final two conditions (.75, 4:2 and .25, 2:4) were not tested.

###########################
### Sessions and blocks ###
###########################

Training/thresholding/staircase blocks were 180 trials long. Training was performed in a single session (Session=0).

In the main experiment, each session was 14 blocks of 50 trials each, for a total of 700 trials per session. There were 7 sessions, corresponding to the 7 conditions, for a total of 4900 trials in the main experiment. There was no more than one session performed per day. Condition order was not fully randomized. Order: 1, then 2/3/4/5 randomized, then 6/7 randomized. Session corresponds to the temporal order the participant experienced.

The first 100 trials from every session were discarded as warmup trials.


################
### Training ###
################

The goal of Training blocks was practice and finding the orientation threshold, which was the only orientation magnitude displayed during the main experiment. Absolute tilt magnitude varied in a series of 3 interleaved 1-up-2-down staircases of 60 trials each to converge on 71% correct. Participants performed multiple blocks until it was determined that performance had plateaued (i.e., learning had stopped). Preliminary thresholds were calculated using the last 10 trials of each staircase. At the end of each block, if none of the three preliminary thresholds were better than the best of the previous block’s preliminary thresholds, then the stopping rule was met. As a result, participants completed a minimum of two blocks and no participant completed more than five blocks. A cumulative Gaussian psychometric function was fit by maximum likelihood to all trials from the final two blocks (360 trials total). The slope parameter was used to calculate the orientation corresponding to 69% correct for an unbiased observer (d'=1; Macmillan, 2005). This orientation was then used for this subject in the main experiment. Thresholds ranged from 0.36 to 0.78 deg, with a mean of 0.59 deg. There is no exact Trial_in_block in Training. On every trial, the displayed stimulus was from a randomly selected staircase until all 3 staircases were completed.


################
### Feedback ###
################

Correct/incorrect feedback was provided on every trial. Points (reward manipulation) were only awarded on correct trials. Confidence responses did not affect points received. Feedback was provided during Training, but no points were awarded.


#################
### Apparatus ###
#################

Dimly lit laboratory. Gamma-corrected CRT monitor (Sony G400, 36x27 cm) with 1280x1024 pixel resolution and 85 Hz refresh rate. Standard Dell keyboard for responses. Chin rest used and viewing distance 57 cm. PsychToolbox 3.0.11 on MATLAB R2014b used for stimulus and responses.


##########################
### Subject population ###
##########################

10 participants (5 female). Age range 22-43 years (mean 27.0). Subject_info column: participant is author, amblyope, or none of these. Other participants recruited from the lab of Michael Landy, and New York University Psychology and Center for Neural Science.


######################
### Reaction times ###
######################

RT_dec and RT_conf measured with PsychToolbox using a standard keyboard. Reaction times were not collected during Training (RT_dec=NaN; RT_conf=NaN).


#############
### Dates ###
#############

Data collected between 10/2017 and 04/2018.


################
### Location ###
################

Data collected in Room 957, Meyer Building, New York University, 6 Washington Pl, New York, NY 10003.
