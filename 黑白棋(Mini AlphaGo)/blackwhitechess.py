import sys
import math
import random
import numpy as np
import board
import copy
reverse_color = {'X': 'O', 'O': 'X'}

class Node:
    def __init__(self, color, board, parent=None):
        self.parent = parent
        self.children=[]
        self.visit_times=0
        self.quality_value = 0.0
        self.color=color
        self.board=board
    def visit_times_add_one(self):
        self.visit_times +=1
    def quality_value_add_n(self,n):
        self.quality_value +=n
    def is_all_expand(self):
        if len(self.children)==len(list(self.board.get_legal_actions(self.color))):
            return True
        else:
            return False
    def add_child(self,child_board):
        child = Node(reverse_color[self.color], child_board, self)
        self.children.append(child)
    def is_terminal(self):
        c = self.board.count('.')
        if c == 0:
            return True
        l1 = list(self.board.get_legal_actions('X'))
        l2 = list(self.board.get_legal_actions('O'))
        if not l1 and not l2:
            return True
        return False
    
class MCTS:
    def __init__(self, color, board, count=50):
        self.color = color
        self.root = Node(color, board)
        self.count = count
        
    def tree_policy(self):#选择子节点的策略
        node=self.root
        while node.is_terminal()==False:
            if node.is_all_expand():
                if len(node.children)==0:
                    break
                node=self.best_child(node,True)
            else:
                sub_node = self.expand(node)
                return sub_node
        return node
            
    def expand(self,node):#顺序扩展得到未扩展的子节点
        board = copy.deepcopy(node.board)
        actions = list(board.get_legal_actions(node.color))
        board._move(actions[len(node.children)], node.color)
        node.add_child(board)
        return node.children[-1]
    
    def best_child(self,node,is_exploration):#若子节点都扩展完了，求UCB值最大的子节点
        best_score=-sys.maxsize
        best_sub_node = []
        for sub_node in node.children:
            if is_exploration:
                C=1/math.sqrt(2.0)
            else:
                C=0.0
            left=sub_node.quality_value/sub_node.visit_times
            right=2.0*math.log(node.visit_times)/sub_node.visit_times
            score=left+C*math.sqrt(right)
            if score>best_score:
                best_sub_node = sub_node
        return best_sub_node
    
    def default_policy(self,node):#返回双方棋子的数量差
            current = copy.deepcopy(node)
            while current.is_terminal()==False:
                actions = list(current.board.get_legal_actions(current.color))
                if len(actions) != 0:
                    current.board._move(random.choice(actions), current.color)
                current.color = reverse_color[current.color] #交换棋手
            return current.board.count(self.color)-current.board.count(reverse_color[self.color])
     
    def backup(self,node,reward):
        while node != None:
            node.visit_times_add_one()
            if(node.color==self.color):
                node.quality_value_add_n(reward)
            else:
                node.quality_value_add_n(-reward)
            node = node.parent

    def monte_carlo_tree_search(self):#蒙特卡洛树搜索总函数
            #computation_budget=1000
            for _ in range(self.count):
                expand_node = self.tree_policy()
                reward = self.default_policy(expand_node)
                self.backup(expand_node,reward)
            best_next_node = self.best_child(self.root,False)
            return best_next_node
        
class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------

        tree = MCTS(self.color, board,count=150)
        node=tree.monte_carlo_tree_search()
        actionboard = node.board
        for c in range(8):
            for r in range(8):
                if ((actionboard[c][r] != board[c][r]) 
                    and (board[c][r] == '.')):
                    return chr(ord('A') + r) + str(c + 1)

        # ------------------------------------------------------------------------

        return action
