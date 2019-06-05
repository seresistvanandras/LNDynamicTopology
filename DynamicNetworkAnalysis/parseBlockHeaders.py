from blockchain import blockexplorer

def getBTCBlockHeaders():
    block = blockexplorer.get_block('000000000000000016f9a2c3e0f4c1245ff24856a79c34806969f5084f410680')
    print(block)

def main():
    getBTCBlockHeaders();

if __name__ == '__main__':
    main()