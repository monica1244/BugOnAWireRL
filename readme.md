## Miniclip Game Interface for Reinforcement Learning
### Tested with a Deep-Q-Network on Bug On a Wire
### Course project for CS 7643: Deep Learning (Prof. Zsolt Kira) @ Georgia Institute of Technology

### Overview
In the course of this project we developed a general purpose deep Reinforcement Learning agent setup, that can be trained to play a host of flash games. We tested our model on the popular miniclip game, Bug on a Wire. Our setup performs the requisite environment interfacing to handle the pre-processing pipeline from raw screenshot images extracted during gameplay as input. This way, our Deep-Q Network (DQN) based agent is able to train itself directly using sensory inputs, in our case, images. We test this by extending our environment+agent setup with minimal changes to play the game Flappy Bird.

We explore two different strategies, namely DQN and DQfD, to train our agent. Initially, we applied an approach which uses an epsilon-greedy DQN based agent. DQN requires a huge training time to achieve good performance, so we explored DQfD, where the agent additionally learns from a set of human performed demonstrations, to quickly learn optimal moves. We perform feature engineering, to generate the state from the raw frames obtained from the flash game. We tested different reward assignments and state parameterizations to bring the model to convergence in a time-efficient fashion.

A complete explanation of the project is available in DL_Project_Report.pdf in this repository.

#### Flash Game Link
[Link to swf](https://www.miniclip.com/games/bug-on-a-wire/en/bug.swf?mc_gamename=Bug+On+A+Wire&mc_hsname=1446&mc_iconBig=bugmedicon.jpg&mc_icon=bugsmallicon.jpg&mc_negativescore=0&mc_players_site=1&mc_scoreistime=0&mc_lowscore=0&mc_width=600&mc_height=300&mc_lang=en&mc_webmaster=0&mc_playerbutton=0&mc_v2=1&loggedin=0&mc_loggedin=0&mc_uid=0&mc_sessid=f78c2dbb92961726d9a87c8f9aa753d2&mc_shockwave=0&mc_gameUrl=%2Fgames%2Fbug-on-a-wire%2Fen%2F&mc_ua=705d28c&mc_geo=us-west-2&mc_geoCode=US&vid=0&vtype=ima&m_vid=1&mc_preroll_check=1&channel=miniclip.preroll&m_channel=miniclip.midroll&s_content=0&mc_plat_id=2&mc_extra=enable_personalized_ads%3D1&mc_image_cdn_path=https%3A%2F%2Favatars.miniclip.com%2F&login_allowed=1&dfp_video_url=https%253A%252F%252Fpubads.g.doubleclick.net%252Fgampad%252Fads%253Fsz%253D600x400%2526iu%253D%252F116850162%252FMiniclip.com_Preroll%2526ciu_szs%2526impl%253Ds%2526gdfp_req%253D1%2526env%253Dvp%2526output%253Dxml_vast2%2526unviewed_position_start%253D1%2526cust_params%253D%2526npa%253D0%2526cust_params%253DgCat%25253Dcategory_13%252526gName%25253Dgame_1446%252526width%25253D600%252526height%25253D300%252526page_domain%25253Dgames%252526gAATF%25253Dgaatf_Y%252526gLanguage%25253Dlanguage_en%252526gPageType%25253Dpagetype_gamepage%252526gDemo1%25253Ddemo1_1%252526gDemo2%25253Ddemo2_2%252526gPageUrl%25253Dhttps%2525253A%2525252F%2525252Fwww.miniclip.com%2525252Fgames%2525252Fbug-on-a-wire%2525252Fen%2525252F%2526url%253D&fn=bug.swf)

[Link to miniclip](https://www.miniclip.com/games/bug-on-a-wire/en/)

#### environment.py
Handles the interfacing with the flash game environment.
Just running `python environment.py` shows the frames as they are sampled.

#### main.py
A hardcoded agent that just plays with right, left and always jumps after a point.

### Windows Setup
1. Windows, because better frame rate.
2. Use conda because pytorch support outside conda is poor.
3. Init conda for powershell- `conda init powershell`.
4. Create a conda environment- `conda create --name conda_env`.
5. Activate the environment- `conda activate conda_env`.
6. Install all packages using `pip install -r requirements.txt`.
7. Install pytorch using conda- `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`.

### Contributors
Nihal Singh @ https://github.com/nihal111<br>
Arindum Roy @ https://github.com/AsliRoy<br>
Roshan Pati @ https://github.com/roshanpati<br>
Monica Gupta @ https://github.com/monica1244
