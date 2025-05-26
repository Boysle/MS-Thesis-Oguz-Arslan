The code works as follows:

Before running the code, users insert the API key they get from [Ballchasing.com](https://ballchasing.com/upload) after signing in.

Then, users will run the code with 'python replay_downloader' command. 

They will need to input 3 variables before the system starts downloading. 

First, users write the group they want to download. Ballcahsing website has a feature where users can organize their replay files using folders called groups. Groups can have other groups as their child but the replay files can only be found on the leaf-child group. Luckily, all the esports replay files can be found on the site organized [here](https://ballchasing.com/groups?creator=76561199225615730) and [here](https://ballchasing.com/groups?creator=76561199022336078).

The link to a group look like this: https://ballchasing.com/group/regional-1-pydu2pstsu
Where the user needs to write only the last part: regional-1-pydu2pstsu

Then, users write the name of the group which will be the name of the folder that holds the downloaded replays.

Lastly, users write the desired directory they want the folder replays to be in. The code is written so that it automatically creates the folders with their group names and puts the replay files in their respective folders.

A log summary of the download process will be placed next to the root folder.

Note: Sometimes the API can be down. Check the status of the API [here](https://ballchasingstatus.com/).