### Ubuntu 16.04

#### Enable 工作區
系統設定值 -> 外觀 -> 運作方式 -> 勾選 **啟用工作區**

#### Shorten command line prompt
Edit .bashrc
```
if [ "$color_prompt" = yes ]; then
    #PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
    PS1='\[\033[01;34m\]\W\[\033[00m\]\$ '
else
    # PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
    PS1='\W\$ '
fi
```

#### Before doing any installations
```
sudo apt-get update
```

#### 倉頡輸入法
```
sudo apt-get install fcitx-table-cangjie3
```
#### Python3
```
sudo apt-get install -y python3-pip python3-setuptools
sudo -H pip3 install --upgrade pip numpy scipy sklearn ipykernel jupyter matplotlib Pillow
```

#### Tensorflow
[reference](https://www.tensorflow.org/install/install_linux)
```
sudo apt-get install libcupti-dev
sudo apt-get install python3-pip python3-dev
sudo -H pip3 install tensorflow
```
Install GPU support version ok but unable to run. ?????
```
sudo -H pip3 install tensorflow-gpu # Python 3.n; GPU support.
```

#### Git
```
sudo apt-get install git
```
**edit ~/.gitconfig**
```
[core]
	editor = vi

[user]
	name = xxx
	email = xxx@xxx.xxx

[credential "https://github.com"]
	username = xxx

[color]
	diff = auto
	status = auto
	branch = auto
	log = auto

[push]
	default = simple

[merge]
	tool = meld
```

#### Support exfat filesystem type

```
sudo apt-get install exfat-utils exfat-fuse
```

#### SSH server (ssh telnet)
```
sudo apt-get install openssh-server
```

#### Chrome
download [Chrome desktop](https://www.google.com.tw/chrome/browser/desktop/index.html)
```
sudo dpkg -i google-chrome-stable_current_amd64.deb
```
Login google account

#### Visual Studio Code
download [Visual Studio Code](https://code.visualstudio.com/download)
```
sudo dpkg -i code_1.14.2-1500506907_amd64.deb
```

#### Slack
download [Slack](https://slack.com/downloads/linux)
```
sudo dpkg -i slack-desktop-2.7.0-amd64.deb
```

#### DOCKER-CE
reference [Docker-CE](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)

#### Change default directories
```
xdg-user-dirs-update --set DESKTOP $HOME/Desktop
xdg-user-dirs-update --set DOCUMENTS $HOME/Documents
xdg-user-dirs-update --set DOWNLOAD $HOME/Download
xdg-user-dirs-update --set MUSIC $HOME/Music
xdg-user-dirs-update --set PICTURES $HOME/Pictures
xdg-user-dirs-update --set PUBLICSHARE $HOME/Publicshare
xdg-user-dirs-update --set TEMPLATES $HOME/Templates
xdg-user-dirs-update --set VIDEOS $HOME/Videos
cp /etc/xdg/user-dirs.conf ~/.config/
```
xdg-user-dirs-update will resett your configuration at each session start up.
Consequently, you need to set enabled=False in your user-dirs.conf file instead of enabled=True. Edit ~/.config/user-dirs.conf
```
#enabled=True
enabled=False
```
When set to False, xdg-user-dirs-update will
not change the XDG user dirs configuration.
If the above doesn't work, you can try edit ~/.config/user-dirs.dirs

#### Make Ubuntu use 'Local' time to solve "Ubuntu/Windows Boot Systems Time Conflicts"
open a terminal and execute the following command
```
timedatectl set-local-rtc 1
```

#### Change the GRUB default boot order
Backup copy of /etc/default/grub and then edit /etc/defalut/grub
```
sudo cp /etc/default/grub /etc/default/grub.bak
sudo gedit /etc/default/grub
```
Find the line that contains
```
GRUB_DEFAULT=0
```
and set it to
```
GRUB_DEFAULT=x
```
where **x** is the index of grub menu item to which you would like to boot to by default. Note that the menu items are zero-indexed. That means that the first item in the list is 0 and that the 3rd item is actually 2. In my case, Windos10 is 3rd item in the list, the line shall be:
```
GRUB_DEFAULT=2
```
Then build the updated grub menu:
```
sudo update-grub
```
