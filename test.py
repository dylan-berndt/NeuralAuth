
from gen import *
import time


class Agent:
    def __init__(self, name):
        self.name = name
        self.preferences = random.choices(Generator.words, k=10)
        print(' '.join(self.preferences))
        self.preferenceEmbedding = Generator.client.embed(self.preferences, query=True)

    def run(self, runs):
        print(f'Running: {runs}')

        context = Generator(9)
        for r in range(runs):
            choices = context.words

            choiceEmbedding = Generator.client.embed(choices, query=False)
            similarity = np.dot(self.preferenceEmbedding, choiceEmbedding.T)[0]

            choice = np.argmax(similarity)

            context.makeChoice(choice)

        context.save(self.name)


if __name__ == '__main__':
    for i in range(40):
        print(i)
        newAgent = Agent(f'agent{i}')
        newAgent.run(random.randint(3, 10))
        print()
        time.sleep(10)




