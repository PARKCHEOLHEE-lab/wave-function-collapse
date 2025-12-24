#!/bin/bash

USERNAME=$USERNAME
SSH="/home/$USERNAME/.ssh/"
IDRSA="$SSH/id_rsa" 

sudo chown -R "$USERNAME:$USERNAME" "$SSH"
sudo chmod 700 "$SSH"
sudo chmod 600 "$IDRSA"

eval "$(ssh-agent -s)"
ssh-add "$IDRSA"