from ollama import Client


def create_client():
    host="localhost:11434"
    client=Client(host)
    return client

if __name__=="__main__":
    client=create_client()

    response=client.chat(model="gemma2:2b",messages=[
        {
            'role':'user',
            'content':'Whos the best football player ever. Generate only a one line answer',
        },
    ])

    print(response['message']['content'])

    another=client.generate(model="gemma2:2b",prompt="What do you think of Logan Sargeant in one line")

    print(another['response'])

    print(response)
