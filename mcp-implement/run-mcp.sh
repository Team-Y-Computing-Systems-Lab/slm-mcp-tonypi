# sudo docker run -it --rm --add-host host.docker.internal:host-gateway --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
sudo docker run -it --rm --add-host host.docker.internal:host-gateway --name n8n -p 5678:5678 -e N8N_SECURE_COOKIE=false -v n8n_data:/home/node/.n8n -v ./shared_folder:/home/node/test docker.n8n.io/n8nio/n8n
# sudo docker run -it --rm --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n



