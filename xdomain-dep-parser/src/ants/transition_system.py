

class TransitionSystem:

    def __init__(self, state: State, action_set: List[Action]):
        self.state = state
        self.action_list = action_list

    def static_oracle(self, instance):
        self.state.initialize(instance)


class InitAction(Action):

    def do(self, state, bsz, max_len, lens):
        state['stack'] = np.zeros((bsz, max_len), dtype='i2')
        assert lens.shape == (bsz, 1)
        state['buffer'] = np.tile(np.arange(max_len, dtype='i2'), (bsz, 1))
        state['buffer'][state['buffer']>lens] = 0
        state['buffer'][:, 0] = lens[:, 0]
        state['tree_a'] = np.zeros((bsz, max_len), dtype='i2')
        state['tree_r'] = np.zeros((bsz, max_len), dtype='i2')
        state['action'] = np.zeros((bsz, 1), dtype='i2')

    def can_do(self):
        raise RuntimeError("InitAction's can_do() should not be executed.")
        pass


class TerminalAction(Action):

    def can_do(self, state):
        # TODO: Use 'tree_a' to simplify judgment.
        return state['buffer'][:, 0]==1 & np.sum(state['stack']>=0, axis=1)==1

    def do(self):
        pass


class ShiftAction(Action):

    def __init__(self, priority):
        super(ShiftAction, self).__init__(priority)

    def can_do(self, state):
        return state['buffer'][:, 0] == 1

    def do(self, state, idx):
        state['stack'][(state['stack']<0)&idx] -= 1
        state['buffer'][(state['buffer']<0)&idx] -= 1
        state['stack'][(state['stack']>0)&idx] += 1
        front = (state['buffer']==1)&idx
        state['stack'][front] = 1
        state['buffer'][front] = -1
        # TODO
        state['action'].append(r)



class LeftArcAction(Action):

    def __init__(self, priority):
        super(LeftArcAction, self).__init__(priority)

    def can_do(self, state):
        return np.sum(state['stack']>0, axis=1)>0 & state['buffer'][:, 0]>0

    def do(self, state, idx, rel):
        # TODO: Do Shift action first
        state['stack'][(state['stack']<0)&idx] -= 1
        state['buffer'][(state['buffer']<0)&idx] -= 1

        top = (state['stack']==1)&idx
        front = (state['buffer']==1)&idx
        state['tree_a'][top] = np.where(front)[1]-np.where(top)[1]
        state['tree_r'][top] = rel
        state['stack'][top] = -1
        state['stack'][(state['stack']>0)&idx] -= 1
        # TODO
        state['action'].append(r)


class RightArcAction(Action):

    def __init__(self, priority):
        super(RightArcAction, self).__init__(priority)

    def can_do(self, state):
        return np.sum(state['stack']==2, axis=1) > 0

    def do(self, state, idx, rel):
        state['stack'][(state['stack']<0)&idx] -= 1
        state['buffer'][(state['buffer']<0)&idx] -= 1

        top2 = (state['stack']==2)&idx
        top = (state['stack']==1)&idx
        state['tree_a'][top] = np.where(top2)[1]-np.where(top)[1]
        state['tree_r'][top] = rel

        # TODO: Do LeftArc action first
        state['stack'][top] = -1
        state['stack'][(state['stack']>0)&idx] -= 1
        # TODO
        state['action'].append(r)


class ArcHybrid:

    def __init__(self):
        self.init_a = InitAction()
        self.terminal_a = TerminalAction()
        self.state = State()
