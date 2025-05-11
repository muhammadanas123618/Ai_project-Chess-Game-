import pygame
import numpy as np
import chess
import chess.pgn
import time
import os
import threading
from typing import Optional, List, Tuple, Dict

class DynamicChessGame:
    def __init__(self):
        pygame.init()
        self.screen_size = 700
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("AI-Enhanced Chess with Dynamic Piece Movements")
        
        # Colors
        self.WHITE = (240, 217, 181)
        self.BLACK = (181, 136, 99)
        self.HIGHLIGHT = (247, 247, 105, 150)
        self.POWERUP_TILE = (106, 168, 79, 150)
        
        # Game state
        self.board = chess.Board()
        self.selected_square = None
        self.valid_moves = []
        self.turn_count = 0
        self.piece_stats = {}  # Track piece movement counts and abilities
        self.powerup_tiles = self.generate_powerup_tiles()
        self.ai_thinking = False
        self.game_over = False
        
        # Create pieces directory if it doesn't exist
        if not os.path.exists("pieces"):
            os.makedirs("pieces")
            self.generate_piece_images()  # Generate images if they don't exist
        
        # Load piece images
        self.piece_images = self.load_piece_images()
        
        # AI settings
        self.ai_depth = 3  # Minimax depth
        self.use_alpha_beta = True

    def generate_piece_images(self):
        """Generate simple piece images if they don't exist"""
        piece_shapes = {
            'pawn': lambda surf, size: pygame.draw.circle(surf, (255, 255, 255), 
                                                        (size//2, size//2), size//3),
            'knight': lambda surf, size: self._draw_knight(surf, size),
            'bishop': lambda surf, size: pygame.draw.polygon(surf, (255, 255, 255), 
                                        [(size//2, size//4), (size//4, 3*size//4), 
                                         (3*size//4, 3*size//4)]),
            'rook': lambda surf, size: pygame.draw.rect(surf, (255, 255, 255), 
                                                       (size//4, size//4, size//2, size//2)),
            'queen': lambda surf, size: pygame.draw.circle(surf, (255, 255, 255), 
                                                        (size//2, size//2), size//3) and
                                       pygame.draw.polygon(surf, (255, 255, 255), 
                                        [(size//2, size//4), (size//4, 3*size//4), 
                                         (3*size//4, 3*size//4)]),
            'king': lambda surf, size: pygame.draw.rect(surf, (255, 255, 255), 
                                                      (size//4, size//4, size//2, size//2)) and
                                      pygame.draw.rect(surf, (255, 255, 255), 
                                                      (size//3, size//8, size//3, size//3))
        }
        
        for color in ['white', 'black']:
            for piece_name in piece_shapes.keys():
                size = self.screen_size // 8
                surf = pygame.Surface((size, size), pygame.SRCALPHA)
                col = (255, 255, 255) if color == 'white' else (50, 50, 50)
                outline_col = (200, 200, 200) if color == 'white' else (0, 0, 0)
                
                # Draw base
                pygame.draw.rect(surf, col, (5, 5, size-10, size-10), 0, 10)
                pygame.draw.rect(surf, outline_col, (5, 5, size-10, size-10), 2, 10)
                
                # Draw piece shape
                piece_shapes[piece_name](surf, size)
                
                # Save image
                pygame.image.save(surf, f"pieces/{color}_{piece_name}.png")

    def _draw_knight(self, surf, size):
        """Helper to draw knight shape"""
        points = [
            (size//2, size//4),
            (3*size//4, size//3),
            (3*size//4, 2*size//3),
            (size//2, 3*size//4),
            (size//4, 2*size//3)
        ]
        pygame.draw.polygon(surf, (255, 255, 255), points)

    def load_piece_images(self) -> Dict[str, pygame.Surface]:
        """Load chess piece images with proper naming"""
        piece_map = {
            'p': 'pawn',
            'n': 'knight',
            'b': 'bishop',
            'r': 'rook',
            'q': 'queen',
            'k': 'king'
        }
        
        images = {}
        
        for color in ['white', 'black']:
            for symbol, piece_name in piece_map.items():
                key = f"{color}_{piece_name}"
                try:
                    images[key] = pygame.image.load(f"pieces/{key}.png")
                    images[key] = pygame.transform.scale(images[key], 
                                                       (self.screen_size//8, self.screen_size//8))
                except:
                    # Fallback: create simple colored rectangles
                    surf = pygame.Surface((self.screen_size//8, self.screen_size//8), pygame.SRCALPHA)
                    col = (255, 255, 255) if color == 'white' else (50, 50, 50)
                    pygame.draw.rect(surf, col, (5, 5, self.screen_size//8-10, self.screen_size//8-10))
                    pygame.draw.rect(surf, (0, 0, 0), (5, 5, self.screen_size//8-10, self.screen_size//8-10), 2)
                    images[key] = surf
                    
        return images
    
    def generate_powerup_tiles(self) -> List[chess.Square]:
        """Generate 4 random power-up tiles on the board"""
        tiles = []
        for _ in range(4):
            tile = np.random.randint(0, 63)
            while tile in tiles:  # Ensure unique tiles
                tile = np.random.randint(0, 63)
            tiles.append(tile)
        return tiles
    
    def draw_board(self):
        """Draw the chess board with pieces and highlights"""
        square_size = self.screen_size // 8
        
        # Draw board squares
        for row in range(8):
            for col in range(8):
                color = self.WHITE if (row + col) % 2 == 0 else self.BLACK
                pygame.draw.rect(self.screen, color, 
                                (col * square_size, row * square_size, square_size, square_size))
                
                # Highlight power-up tiles
                square = chess.square(col, 7-row)
                if square in self.powerup_tiles:
                    highlight = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
                    highlight.fill(self.POWERUP_TILE)
                    self.screen.blit(highlight, (col * square_size, row * square_size))
                
                # Highlight selected square
                if self.selected_square == square:
                    highlight = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
                    highlight.fill(self.HIGHLIGHT)
                    self.screen.blit(highlight, (col * square_size, row * square_size))
                
                # Highlight valid moves
                if square in self.valid_moves:
                    pygame.draw.circle(self.screen, self.HIGHLIGHT, 
                                     (col * square_size + square_size//2, 
                                      row * square_size + square_size//2), 
                                     square_size//4)
        
        # Draw pieces
        piece_map = {
            'p': 'pawn',
            'n': 'knight',
            'b': 'bishop',
            'r': 'rook',
            'q': 'queen',
            'k': 'king'
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                col = chess.square_file(square)
                row = 7 - chess.square_rank(square)
                color = "white" if piece.color == chess.WHITE else "black"
                piece_type = piece_map[piece.symbol().lower()]
                self.screen.blit(self.piece_images[f"{color}_{piece_type}"], 
                               (col * square_size, row * square_size))
        
        # Display turn information
        font = pygame.font.SysFont("Arial", 20)
        turn_text = f"Turn: {self.turn_count} | {'White' if self.board.turn == chess.WHITE else 'Black'} to move"
        text_surface = font.render(turn_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))
        
        if self.ai_thinking:
            ai_text = font.render("AI is thinking...", True, (0, 0, 0))
            self.screen.blit(ai_text, (10, 40))
            
        if self.game_over:
            result = self.board.result()
            game_over_text = font.render(f"Game Over: {result}", True, (255, 0, 0))
            self.screen.blit(game_over_text, (self.screen_size//2 - 100, 10))

    def handle_click(self, pos):
        """Handle mouse clicks for piece selection and movement"""
        if self.game_over or self.ai_thinking or self.board.turn == chess.BLACK:
            return
            
        col = pos[0] // (self.screen_size // 8)
        row = pos[1] // (self.screen_size // 8)
        square = chess.square(col, 7 - row)
        
        # If a piece is already selected, try to move it
        if self.selected_square is not None:
            if square in self.valid_moves:
                self.make_move(self.selected_square, square)
                self.selected_square = None
                self.valid_moves = []
                
                # AI makes a move after player
                if not self.board.is_game_over() and self.board.turn == chess.BLACK:
                    self.ai_move()
            elif self.board.piece_at(square) and self.board.piece_at(square).color == self.board.turn:
                # Select a different piece
                self.selected_square = square
                self.valid_moves = self.get_valid_moves(square)
            else:
                # Clicked on invalid square, deselect
                self.selected_square = None
                self.valid_moves = []
        else:
            # Select a piece if it's the player's turn
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.valid_moves = self.get_valid_moves(square)
    
    def get_valid_moves(self, square: chess.Square) -> List[chess.Square]:
        """Get all valid moves for a piece, including dynamic movement options"""
        piece = self.board.piece_at(square)
        if not piece:
            return []
            
        # Standard moves
        valid_moves = []
        for move in self.board.legal_moves:
            if move.from_square == square:
                valid_moves.append(move.to_square)
                
        # Dynamic movement additions
        piece_key = f"{'white' if piece.color == chess.WHITE else 'black'}_{chess.square_name(square)}"
        
        # Track piece stats if not already tracked
        if piece_key not in self.piece_stats:
            self.piece_stats[piece_key] = {
                'move_count': 0,
                'capture_count': 0,
                'special_active': False
            }
            
        # Knight dynamic movement - after 3 moves, can jump two pieces
        if piece.piece_type == chess.KNIGHT and self.piece_stats[piece_key]['move_count'] >= 3:
            # Add extended knight moves
            directions = [(2, 1), (1, 2), (-1, 2), (-2, 1), 
                         (-2, -1), (-1, -2), (1, -2), (2, -1),
                         (3, 1), (1, 3), (-1, 3), (-3, 1),
                         (-3, -1), (-1, -3), (1, -3), (3, -1),
                         (3, 2), (2, 3), (-2, 3), (-3, 2),
                         (-3, -2), (-2, -3), (2, -3), (3, -2)]
            
            current_file = chess.square_file(square)
            current_rank = chess.square_rank(square)
            
            for df, dr in directions:
                new_file = current_file + df
                new_rank = current_rank + dr
                
                if 0 <= new_file < 8 and 0 <= new_rank < 8:
                    new_square = chess.square(new_file, new_rank)
                    if not self.board.piece_at(new_square) or self.board.piece_at(new_square).color != piece.color:
                        valid_moves.append(new_square)
                        
        # Bishop dynamic movement - after capturing, can move like queen for one turn
        if piece.piece_type == chess.BISHOP and self.piece_stats[piece_key]['special_active']:
            # Add queen-like moves
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1),
                         (1, 1), (1, -1), (-1, 1), (-1, -1)]
            
            current_file = chess.square_file(square)
            current_rank = chess.square_rank(square)
            
            for df, dr in directions:
                for i in range(1, 8):
                    new_file = current_file + df * i
                    new_rank = current_rank + dr * i
                    
                    if 0 <= new_file < 8 and 0 <= new_rank < 8:
                        new_square = chess.square(new_file, new_rank)
                        target_piece = self.board.piece_at(new_square)
                        
                        if not target_piece:
                            valid_moves.append(new_square)
                        else:
                            if target_piece.color != piece.color:
                                valid_moves.append(new_square)
                            break
                    else:
                        break
        
        # Power-up tile effects
        if square in self.powerup_tiles:
            current_file = chess.square_file(square)
            current_rank = chess.square_rank(square)
            
            # Increased movement range - add all squares within 2 steps
            for df in range(-2, 3):
                for dr in range(-2, 3):
                    if df == 0 and dr == 0:
                        continue
                        
                    new_file = current_file + df
                    new_rank = current_rank + dr
                    
                    if 0 <= new_file < 8 and 0 <= new_rank < 8:
                        new_square = chess.square(new_file, new_rank)
                        if not self.board.piece_at(new_square) or self.board.piece_at(new_square).color != piece.color:
                            if new_square not in valid_moves:
                                valid_moves.append(new_square)
        
        return list(set(valid_moves))  # Remove duplicates
    
    def make_move(self, from_square: chess.Square, to_square: chess.Square):
        """Execute a move and handle dynamic rule updates"""
        piece = self.board.piece_at(from_square)
        if not piece:
            return
            
        piece_key = f"{'white' if piece.color == chess.WHITE else 'black'}_{chess.square_name(from_square)}"
        
        # Check for captured piece before making the move
        captured_piece = self.board.piece_at(to_square)
        
        # Create move (handles promotion automatically)
        move = None
        for legal_move in self.board.legal_moves:
            if legal_move.from_square == from_square and legal_move.to_square == to_square:
                move = legal_move
                break
                
        if not move:
            # Might be a dynamic move - create a new move
            move = chess.Move(from_square, to_square)
            
            # Handle promotion
            if piece.piece_type == chess.PAWN and chess.square_rank(to_square) in [0, 7]:
                move.promotion = chess.QUEEN  # Default to queen promotion
        
        # Execute the move
        self.board.push(move)
        self.turn_count += 1
        
        # Update piece stats
        if piece_key in self.piece_stats:
            self.piece_stats[piece_key]['move_count'] += 1
            
            # Check if capture occurred
            if captured_piece and captured_piece.color != piece.color:
                self.piece_stats[piece_key]['capture_count'] += 1
                
                # Bishop special ability activation
                if piece.piece_type == chess.BISHOP:
                    self.piece_stats[piece_key]['special_active'] = True
        else:
            self.piece_stats[piece_key] = {
                'move_count': 1,
                'capture_count': 1 if captured_piece and captured_piece.color != piece.color else 0,
                'special_active': piece.piece_type == chess.BISHOP and captured_piece and captured_piece.color != piece.color
            }
        
        # Check for game over
        if self.board.is_game_over():
            self.game_over = True
    
    def ai_move(self):
        """AI makes a move using Minimax with Alpha-Beta pruning"""
        self.ai_thinking = True
        self.draw_board()
        pygame.display.flip()
        
        def ai_thread():
            best_move = self.find_best_move()
            self.make_move(best_move.from_square, best_move.to_square)
            self.ai_thinking = False
            
        threading.Thread(target=ai_thread).start()
    
    def find_best_move(self) -> chess.Move:
        """Find the best move using Minimax algorithm with Alpha-Beta pruning"""
        best_move = None
        best_value = -float('inf')
        
        for move in self.board.legal_moves:
            self.board.push(move)
            move_value = self.minimax(self.ai_depth - 1, -float('inf'), float('inf'), False)
            self.board.pop()
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
                
        return best_move if best_move else list(self.board.legal_moves)[0]
    
    def minimax(self, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
        """Minimax algorithm with Alpha-Beta pruning"""
        if depth == 0 or self.board.is_game_over():
            return self.evaluate_board()
            
        if maximizing_player:
            max_eval = -float('inf')
            for move in self.board.legal_moves:
                self.board.push(move)
                eval = self.minimax(depth - 1, alpha, beta, False)
                self.board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if self.use_alpha_beta and beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.board.legal_moves:
                self.board.push(move)
                eval = self.minimax(depth - 1, alpha, beta, True)
                self.board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if self.use_alpha_beta and beta <= alpha:
                    break
            return min_eval
    
    def evaluate_board(self) -> float:
        """Evaluate the board position with custom heuristics"""
        if self.board.is_checkmate():
            return float('inf') if self.board.turn == chess.BLACK else -float('inf')
        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves():
            return 0
            
        # Material score
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King value handled separately
        }
        
        white_score = 0
        black_score = 0
        
        # Material evaluation
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_score += value
                else:
                    black_score += value
        
        # Mobility score
        white_mobility = len(list(self.board.legal_moves))
        self.board.push(chess.Move.null())  # Switch turn to calculate opponent mobility
        black_mobility = len(list(self.board.legal_moves))
        self.board.pop()
        
        white_score += white_mobility * 0.1
        black_score += black_mobility * 0.1
        
        # Piece activity bonuses
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_key = f"{'white' if piece.color == chess.WHITE else 'black'}_{chess.square_name(square)}"
                
                # Bonus for pieces on power-up tiles
                if square in self.powerup_tiles:
                    if piece.color == chess.WHITE:
                        white_score += 0.5
                    else:
                        black_score += 0.5
                
                # Bonus for pieces with dynamic abilities
                if piece_key in self.piece_stats:
                    if piece.piece_type == chess.KNIGHT and self.piece_stats[piece_key]['move_count'] >= 3:
                        if piece.color == chess.WHITE:
                            white_score += 0.3
                        else:
                            black_score += 0.3
                    if piece.piece_type == chess.BISHOP and self.piece_stats[piece_key]['special_active']:
                        if piece.color == chess.WHITE:
                            white_score += 0.5
                        else:
                            black_score += 0.5
        
        # King safety evaluation
        white_king_square = self.board.king(chess.WHITE)
        black_king_square = self.board.king(chess.BLACK)
        
        white_king_safety = self.evaluate_king_safety(white_king_square, chess.WHITE)
        black_king_safety = self.evaluate_king_safety(black_king_square, chess.BLACK)
        
        white_score += white_king_safety
        black_score += black_king_safety
        
        return white_score - black_score if self.board.turn == chess.WHITE else black_score - white_score
    
    def evaluate_king_safety(self, king_square: chess.Square, color: chess.Color) -> float:
        """Evaluate king safety based on pawn shield and enemy attacks"""
        if not king_square:
            return 0
            
        safety_score = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Pawn shield evaluation
        for df in [-1, 0, 1]:
            for dr in [1, 2] if color == chess.WHITE else [-1, -2]:
                file = king_file + df
                rank = king_rank + dr
                
                if 0 <= file < 8 and 0 <= rank < 8:
                    square = chess.square(file, rank)
                    piece = self.board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        safety_score += 0.2 if dr == 1 or dr == -1 else 0.1
        
        # Enemy attack evaluation
        enemy_color = not color
        attack_score = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == enemy_color:
                # Simple attack count (could be enhanced with actual attack calculation)
                if square in self.powerup_tiles:
                    attack_score += 0.3
                else:
                    attack_score += 0.1
        
        return safety_score - attack_score * 0.5
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
            
            self.draw_board()
            pygame.display.flip()
            clock.tick(60)
            
        pygame.quit()

if __name__ == "__main__":
    game = DynamicChessGame()
    game.run()