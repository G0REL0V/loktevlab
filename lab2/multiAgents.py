# -*- coding: utf-8 -*-
# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        Вам не нужно изменять этот метод, но вы можете сделать это, если хотите.

        getAction выбирает лучшее действие в соответствии с оценочной функцией

        getAction берет GameState и возвращает некоторое Directions.X, где
        X - одно из значений {North, South, West, East, Stop}
        """
        # Собираем возможные дейстия и следующие состояния
        legalMoves = gameState.getLegalActions()

        # Выбираем лучшее действие
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Берем любое из лучших

        "Добавьте что-то еще, если хотите"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Реализуйте здесь лучшую оценочную функцию

        Оценочная функция берет текущее и предполагаемое следующее состояние
        GameStates (pacman.py) и возвращает число (чем больше число, тем лучше).

        Код ниже содержит полезную информацию о состоянии, такую как оставшуюся еду
        (newFood) и позицию пакмена после движения (newPos).
        newScaredTimes содержит число шагов, в течении которых каждый призрак
        останется напуганным, поскольку пакмен съел гранулу силы.

        Выведите эти переменные, чтобы посмотреть, что в них находится, и затем
        комбинируйте их для создания великолепной оценочной функции.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        def sum_food_proximity(cur_pos, food_positions, norm=False):
            food_distances = []
            for food in food_positions:
                food_distances.append(util.manhattanDistance(food, cur_pos))
            if norm:
                return normalize(sum(food_distances) if sum(food_distances) > 0 else 1)
            else:
                return sum(food_distances) if sum(food_distances) > 0 else 1

        score = successorGameState.getScore()
        def ghost_stuff(cur_pos, ghost_states, radius, scores):
            num_ghosts = 0
            for ghost in ghost_states:
                if util.manhattanDistance(ghost.getPosition(), cur_pos) <= radius:
                    scores -= 30
                    num_ghosts += 1
            return scores

        def food_stuff(cur_pos, food_pos, cur_score):
            new_food = sum_food_proximity(cur_pos, food_pos)
            cur_food = sum_food_proximity(currentGameState.getPacmanPosition(), currentGameState.getFood().asList())
            new_food = 1/new_food
            cur_food = 1/cur_food
            if new_food > cur_food:
                cur_score += (new_food - cur_food) * 3
            else:
                cur_score -= 20

            next_food_dist = closest_dot(cur_pos, food_pos)
            cur_food_dist = closest_dot(currentGameState.getPacmanPosition(), currentGameState.getFood().asList())
            if next_food_dist < cur_food_dist:
                cur_score += (next_food_dist - cur_food_dist) * 3
            else:
                cur_score -= 20
            return cur_score

        def closest_dot(cur_pos, food_pos):
            food_distances = []
            for food in food_pos:
                food_distances.append(util.manhattanDistance(food, cur_pos))
            return min(food_distances) if len(food_distances) > 0 else 1


        def normalize(distance, layout):
            return distance

        return food_stuff(newPos, newFood.asList(), ghost_stuff(newPos, newGhostStates, 2, score))


def scoreEvaluationFunction(currentGameState):
    """
      Эта оченочная функция просто возвращает количество очков для состояния.
      Количество очков отображается в графическом интерфейсе.

      Эта оценочная функция должна быть использована с

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      Этот класс предоставляет некоторые общие элементы для всех поисковых
      мультиагентов. Любой из определенных здесь методов будет доступен в классах
      MinimaxPacmanAgent, AlphaBetaPacmanAgent и ExpectimaxPacmanAgent.

      Вам не нужно ничего сдесь изменять, но вы можете добавить любую дополнительную
      функциональность для ваших поисковых агентов. Только не удаляйте ничего, пожалуйста ;)

      Замечание: это абстрактный класс - нельзя создавать его объект. Он только частично
      реализован, и требует расширения. Agent (в game.py) - тоже абстрактный класс.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Ваш минимакс агент (задание 2, вариант 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        PACMAN = 0
        def max_agent(state, depth):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            score = best_score
            best_action = Directions.STOP
            for action in actions:
                score = exp_agent(state.generateSuccessor(PACMAN, action), depth, 1)
                if score > best_score:
                    best_score = score
                    best_action = action
            if depth == 0:
                return best_action
            else:
                return best_score

        def exp_agent(state, depth, ghost):
            if state.isLose() or state.isWin():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
                # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                if next_ghost == PACMAN: # We are on the last ghost and it will be Pacman's turn next.
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else:
                        score = max_agent(state.generateSuccessor(ghost, action), depth + 1)
                else:
                    score = exp_agent(state.generateSuccessor(ghost, action), depth, next_ghost)
                if score < best_score:
                    best_score = score
            return best_score
        return max_agent(gameState, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Ваш минимакс агент с альфа-бета отсечением (задание 2, вариант 2)
    """

    def getAction(self, gameState):
        """
          Возвращает действие minimax, используя self.depth и self.evaluationFunction
        """
        PACMAN = 0
        def max_agent(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            score = best_score
            best_action = Directions.STOP
            for action in actions:
                score = min_agent(state.generateSuccessor(PACMAN, action), depth, 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if best_score > beta:
                    return best_score
            if depth == 0:
                return best_action
            else:
                return best_score

        def min_agent(state, depth, ghost, alpha, beta):
            if state.isLose() or state.isWin():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
                # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                if next_ghost == PACMAN: # We are on the last ghost and it will be Pacman's turn next.
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else:
                        score = max_agent(state.generateSuccessor(ghost, action), depth + 1, alpha, beta)
                else:
                    score = min_agent(state.generateSuccessor(ghost, action), depth, next_ghost, alpha, beta)
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)
                if best_score < alpha:
                    return best_score
            return best_score
        return max_agent(gameState, 0, float("-inf"), float("inf"))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Ваш агент expectimax (задание 2, вариант 3)
    """

    def getAction(self, gameState):
        """
          Возвращает действие expectimax, используя self.depth и self.evaluationFunction

          Считайте, что все призраки выбирают случайные ходы из доступных им в данный момент.
        """
        PACMAN = 0
        def max_agent(state, depth):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            score = best_score
            best_action = Directions.STOP
            for action in actions:
                score = min_agent(state.generateSuccessor(PACMAN, action), depth, 1)
                if score > best_score:
                    best_score = score
                    best_action = action
            if depth == 0:
                return best_action
            else:
                return best_score

        def min_agent(state, depth, ghost):
            if state.isLose():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
                # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                prob = 1.0/len(actions)
                if next_ghost == PACMAN: # We are on the last ghost and it will be Pacman's turn next.
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                        score += prob * score
                    else:
                        score = max_agent(state.generateSuccessor(ghost, action), depth + 1)
                        score += prob * score
                else:
                    score = min_agent(state.generateSuccessor(ghost, action), depth, next_ghost)
                    score += prob * score
            return score
        return max_agent(gameState, 0)

def betterEvaluationFunction(currentGameState):
    """
      Если хотите, напишите агента по своему собственному алгоритму, и опишите
      логику его работы здесь.
    """
    def closest_dot(cur_pos, food_pos):
        food_distances = []
        for food in food_pos:
            food_distances.append(util.manhattanDistance(food, cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1

    def closest_ghost(cur_pos, ghosts):
        food_distances = []
        for food in ghosts:
            food_distances.append(util.manhattanDistance(food.getPosition(), cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1


    def ghost_stuff(cur_pos, ghost_states, radius, scores):
        num_ghosts = 0
        for ghost in ghost_states:
            if util.manhattanDistance(ghost.getPosition(), cur_pos) <= radius:
                scores -= 30
                num_ghosts += 1
        return scores

    def food_stuff(cur_pos, food_positions):
        food_distances = []
        for food in food_positions:
            food_distances.append(util.manhattanDistance(food, cur_pos))
        return sum(food_distances)

    def num_food(cur_pos, food):
        return len(food)

    pacman_pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()

    score = score * 2 if closest_dot(pacman_pos, food) < closest_ghost(pacman_pos, ghosts) + 3 else score
    score -= .35 * food_stuff(pacman_pos, food)
    return score

# Аббревиатура
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.
          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        PACMAN = 0

        def maxi_agent(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            score = best_score
            best_action = Directions.STOP
            for action in actions:
                score = expecti_agent(state.generateSuccessor(PACMAN, action), depth, 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
            if depth == 0:
                return best_action
            else:
                return best_score

        def expecti_agent(state, depth, ghost, alpha, beta):
            if state.isLose():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
                # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                prob = .8
                if next_ghost == PACMAN: # We are on the last ghost and it will be Pacman's turn next.
                    if depth == 3:
                        score = contestEvaluationFunc(state.generateSuccessor(ghost, action))
                        score += prob * score
                    else:
                        score = max_agent(state.generateSuccessor(ghost, action), depth + 1, alpha, beta)
                        score += prob * score
                else:
                    score = expecti_agent(state.generateSuccessor(ghost, action), depth, next_ghost, alpha, beta)
                    score += (1-prob) * score
            return score

        def max_agent(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            score = best_score
            best_action = Directions.STOP
            for action in actions:
                score = min_agent(state.generateSuccessor(PACMAN, action), depth, 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if best_score > beta:
                    return best_score
            if depth == 0:
                return best_action
            else:
                return best_score

        def min_agent(state, depth, ghost, alpha, beta):
            if state.isLose() or state.isWin():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
                # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                if next_ghost == PACMAN: # We are on the last ghost and it will be Pacman's turn next.
                    if depth == 3:
                        score = contestEvaluationFunc(state.generateSuccessor(ghost, action))
                    else:
                        score = max_agent(state.generateSuccessor(ghost, action), depth + 1, alpha, beta)
                else:
                    score = min_agent(state.generateSuccessor(ghost, action), depth, next_ghost, alpha, beta)
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)
                if best_score < alpha:
                    return best_score
            return best_score
        return maxi_agent(gameState, 0, float("-inf"), float("inf"))

def contestEvaluationFunc(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    def closest_dot(cur_pos, food_pos):
        food_distances = []
        for food in food_pos:
            food_distances.append(util.manhattanDistance(food, cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1

    def closest_ghost(cur_pos, ghosts):
        food_distances = []
        for food in ghosts:
            food_distances.append(util.manhattanDistance(food.getPosition(), cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1


    def ghost_stuff(cur_pos, ghost_states, radius, scores):
        num_ghosts = 0
        for ghost in ghost_states:
            if util.manhattanDistance(ghost.getPosition(), cur_pos) <= radius:
                scores -= 30
                num_ghosts += 1
        return scores

    def food_stuff(cur_pos, food_positions):
        food_distances = []
        for food in food_positions:
            food_distances.append(util.manhattanDistance(food, cur_pos))
        return sum(food_distances)

    def num_food(cur_pos, food):
        return len(food)

    def closest_capsule(cur_pos, caps_pos):
        capsule_distances = []
        for caps in caps_pos:
            capsule_distances.append(util.manhattanDistance(caps, cur_pos))
        return min(capsule_distances) if len(capsule_distances) > 0 else 9999999

    def scaredghosts(ghost_states, cur_pos, scores):
        scoreslist = []
        for ghost in ghost_states:
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 4:
                scoreslist.append(scores + 50)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 3:
                scoreslist.append(scores + 60)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 2:
                scoreslist.append(scores + 70)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 1:
                scoreslist.append(scores + 90)
            #if ghost.scaredTimer > 0 and util.manhattanDistance(ghost.getPosition(), cur_pos) < 1:
 #              scoreslist.append(scores + 100)
        return max(scoreslist) if len(scoreslist) > 0 else scores

    def ghostattack(ghost_states, cur_pos, scores):
        scoreslist = []
        for ghost in ghost_states:
            if ghost.scaredTimer == 0:
                scoreslist.append(scores - util.manhattanDistance(ghost.getPosition(), cur_pos) - 10)
        return max(scoreslist) if len(scoreslist) > 0 else scores

    def scoreagent(cur_pos, food_pos, ghost_states, caps_pos, score):
        if closest_capsule(cur_pos, caps_pos) < closest_ghost(cur_pos, ghost_states):
            return score + 40
        if closest_dot(cur_pos, food_pos) < closest_ghost(cur_pos, ghost_states) + 3:
            return score + 20
        if closest_capsule(cur_pos, caps_pos) < closest_dot(cur_pos, food_pos) + 3:
            return score + 30
        else:
            return score


    capsule_pos = currentGameState.getCapsules()
    pacman_pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()

    #score = score * 2 if closest_dot(pacman_pos, food) < closest_ghost(pacman_pos, ghosts) + 3 else score
    #score = score * 1.5 if closest_capsule(pacman_pos, capsule_pos) < closest_dot(pacman_pos, food) + 4 else score
    score = scoreagent(pacman_pos, food, ghosts, capsule_pos, score)
    score = scaredghosts(ghosts, pacman_pos, score)
    score = ghostattack(ghosts, pacman_pos, score)
    score -= .35 * food_stuff(pacman_pos, food)
    return score