#sudo -E nohup python3 bot_chatgpt_agents.py 2>&1 1> output.log &
nohup uvicorn conversational_api:app --host 0.0.0.0 --port 9999 --reload 2>&1 1> outputAPI.log & 

