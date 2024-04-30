class Chess:
    def __init__(self):
        self.bitboards()
    def bitboards(self):
        self.white_pawns = 0x000000000000FF00
        self.black_pawns = 0x00FF000000000000
        self.white_knights = 0x0000000000000042
        self.black_knights = 0x4200000000000000
        self.white_bishops = 0x0000000000000024
        self.black_bishops = 0x2400000000000000
        self.white_rooks = 0x0000000000000081
        self.black_rooks = 0x8100000000000000
        self.white_queens = 0x0000000000000008
        self.black_queens = 0x0800000000000000
        self.white_king = 0x0000000000000010
        self.black_king = 0x1000000000000000
        self.white_pieces = self.white_pawns | self.white_knights | self.white_bishops | self.white_rooks | self.white_queens | self.white_king
        self.black_pieces = self.black_pawns | self.black_knights | self.black_bishops | self.black_rooks | self.black_queens | self.black_king
        self.all_pieces = self.white_pieces | self.black_pieces
        self.current_player = True
        self.castling_rights = 0b1111