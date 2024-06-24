from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from openai import OpenAI
import threading
import time
import random
import json
import csv

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": "*"}})

# Global token limit
TOKEN_LIMIT = 2048

class LMStudioAgent:
    def __init__(self, name, api_url, api_key, model, temperature=0.7, starting_prompt=""):
        self.name = name
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.starting_prompt = starting_prompt
        self.history = [
            {"role": "system", "content": starting_prompt}
        ]
        self.running = False
        self.thread = None
        self.waiting_for_user_input = False

    def reset_history(self):
        self.history = [
            {"role": "system", "content": self.starting_prompt}
        ]

    def respond_once(self, message):
        if not self.waiting_for_user_input:
            self.history.append({"role": "user", "content": message})

            context_tokens = int(TOKEN_LIMIT * 0.5)
            context = self._get_context(context_tokens)

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=context,
                temperature=self.temperature,
                stream=True,
            )

            new_message = {"role": "assistant", "content": ""}
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    new_message["content"] += chunk.choices[0].delta.content

            self.history.append(new_message)

            socketio.emit('new_message', {'role': self.name, 'content': new_message["content"]})

            try:
                self.save_message_to_csv(new_message["content"])
            except Exception as e:
                print(f"Error saving message to CSV: {e}")

            self.wait_for_user_input()

            return new_message["content"]

    def _get_context(self, context_tokens):
        context = []
        total_tokens = 0
        for message in reversed(self.history):
            message_tokens = len(message["content"].split())
            if total_tokens + message_tokens > context_tokens:
                break
            context.insert(0, message)
            total_tokens += message_tokens
        return context

    def save_message_to_csv(self, message):
        with open('agent_responses.csv', mode='a', encoding="utf-8", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.name, message])

    def start_conversation(self, topic):
        self.running = True
        self.reset_history()
        self.history.append({"role": "user", "content": topic})
        self.thread = threading.Thread(target=self.run_conversation)
        self.thread.start()

    def stop_conversation(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def run_conversation(self):
        while self.running:
            if self.waiting_for_user_input:
                time.sleep(1)
                continue

            message = self.history[-1]["content"]
            self.respond_once(message)

        print(f"{self.name} conversation ended.")

    def wait_for_user_input(self):
        self.waiting_for_user_input = True

    def receive_user_input(self):
        self.waiting_for_user_input = False

agents = []

def load_agents_from_config(config_file):
    global agents
    with open(config_file, 'r') as f:
        config = json.load(f)
    agents = []
    for agent_config in config:
        agent = LMStudioAgent(
            name=agent_config["name"],
            api_url=agent_config["api_url"],
            api_key=agent_config["api_key"],
            model=agent_config["model"],
            temperature=agent_config["temperature"],
            starting_prompt=agent_config["starting_prompt"]
        )
        agents.append(agent)
    return agents

load_agents_from_config('agents_config.json')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_conversation')
def handle_start_conversation(data):
    topic = data.get('topic')
    selected_agent = data.get('agent')

    if topic:
        if selected_agent == 'all':
            for agent in agents:
                agent.start_conversation(topic)
        else:
            agent = next((agent for agent in agents if agent.name == selected_agent), None)
            if agent:
                agent.start_conversation(topic)

        socketio.emit('new_message', {'role': 'system', 'content': f"Starting conversation on topic: {topic} (with {selected_agent})"})
    else:
        print("Error: Missing 'topic' in data.")

@socketio.on('stop_conversation')
def handle_stop_conversation():
    for agent in agents:
        agent.stop_conversation()

    socketio.emit('new_message', {'role': 'system', 'content': "Conversation stopped by user"})

if __name__ == '__main__':
    socketio.run(app, debug=True)