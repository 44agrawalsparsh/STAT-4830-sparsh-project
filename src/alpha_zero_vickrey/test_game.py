from basic_game.basic_auction import BasicAuctionGame


game = BasicAuctionGame()

cur_state = game.getInitState()

while True:
    print(cur_state)

    def get_input():
        try:
            return float(input("Enter action:"))
        except:
            return get_input()

    cur_state = game.getNextState(cur_state, get_input())