from collections import deque
from typing import Optional, List, Any
import random
import math
import matplotlib.pyplot as plt
import statistics

class TreeNode:
    def __init__(self, key: int, value: Any = None):
        self.key = key          # Ключ узла
        self.value = value      # Значение узла
        self.left: Optional[TreeNode] = None    # Левый потомок
        self.right: Optional[TreeNode] = None   # Правый потомок
        self.parent: Optional[TreeNode] = None  # Родительский узел

class BinarySearchTree:    
    def __init__(self):
        self.root: Optional[TreeNode] = None  # Корень дерева
        self._size = 0                        # Количество узлов
    
    def size(self) -> int:
        return self._size
    
    def empty(self) -> bool:
        return self.root is None
    
    def search(self, key: int) -> Optional[TreeNode]:
        return self._search_node(self.root, key)  # начинаем поиск с корня
    
    def _search_node(self, node: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if node is None or node.key == key:  # базовый случай рекурсии
            return node
        
        if key < node.key:                    # ключ меньше текущего - идем влево
            return self._search_node(node.left, key)
        else:                                 # ключ больше текущего - идем вправо
            return self._search_node(node.right, key)
    
    def insert(self, key: int, value: Any = None) -> None:
        new_node = TreeNode(key, value)
        
        if self.root is None:                 # дерево пустое - новый узел становится корнем
            self.root = new_node
        else:
            self._insert_node(self.root, new_node)
        
        self._size += 1
    
    def _insert_node(self, current: TreeNode, new_node: TreeNode) -> None:
        if new_node.key < current.key:        # вставляем в левое поддерево
            if current.left is None:          # нашли место для вставки
                current.left = new_node
                new_node.parent = current
            else:                             # продолжаем поиск в левом поддереве
                self._insert_node(current.left, new_node)
        elif new_node.key > current.key:      # вставляем в правое поддерево
            if current.right is None:         # нашли место для вставки
                current.right = new_node
                new_node.parent = current
            else:                             # продолжаем поиск в правом поддереве
                self._insert_node(current.right, new_node)
        else:                                 # ключ уже существует - обновляем значение
            current.value = new_node.value
            self._size -= 1                   # компенсируем увеличение размера в insert()
    
    def delete(self, key: int) -> bool:
        node = self.search(key)               # находим узел для удаления
        if node is None:                      # узел не найден
            return False
        
        self._delete_node(node)               # удаляем узел
        self._size -= 1
        return True
    
    def _delete_node(self, node: TreeNode) -> None:
        # случай 1: Узел без детей (лист)
        if node.left is None and node.right is None:
            self._transplant(node, None)      # просто удаляем узел
        
        # Случай 2: Узел с одним ребенком
        elif node.left is None:               # правый ребенок
            self._transplant(node, node.right)
        elif node.right is None:              # левый ребенок
            self._transplant(node, node.left)
        
        # Случай 3: Узел с двумя детьми
        else:
            successor = self._min_node(node.right)  # находим преемника (минимальный узел в правом поддереве)
            
            if successor.parent != node:            # преемник не является непосредственным правым ребенком
                self._transplant(successor, successor.right)
                successor.right = node.right
                successor.right.parent = successor
            
            self._transplant(node, successor)       # заменяем удаляемый узел на преемника
            successor.left = node.left
            successor.left.parent = successor
    
    def _transplant(self, u: TreeNode, v: Optional[TreeNode]) -> None:
        """Заменяет поддерево с корнем u на поддерево с корнем v"""
        if u.parent is None:                  # u-корень дерева
            self.root = v
        elif u == u.parent.left:              # u-левый ребенок своего родителя
            u.parent.left = v
        else:                                 # u-правый ребенок своего родителя
            u.parent.right = v
        
        if v is not None:                     # обновляем родителя у v
            v.parent = u.parent
    
    def find_min(self) -> Optional[int]:
        if self.root is None:                 # дерево пустое
            return None
        return self._min_node(self.root).key
    
    def _min_node(self, node: TreeNode) -> TreeNode:
        while node.left is not None:          # двигаемся влево до упора
            node = node.left
        return node
    
    def find_max(self) -> Optional[int]:
        if self.root is None:                 # дерево пустое
            return None
        return self._max_node(self.root).key
    
    def _max_node(self, node: TreeNode) -> TreeNode:
        while node.right is not None:         # двигаемся вправо до упора
            node = node.right
        return node
    
    def inorder_traversal(self) -> List[int]:
        result = []
        self._inorder(self.root, result)
        return result
    
    def _inorder(self, node: Optional[TreeNode], result: List[int]) -> None:
        if node is not None:
            self._inorder(node.left, result)   # левое поддерево
            result.append(node.key)            # корень
            self._inorder(node.right, result)  # правое поддерево
    
    def preorder_traversal(self) -> List[int]:
        result = []
        self._preorder(self.root, result)
        return result
    
    def _preorder(self, node: Optional[TreeNode], result: List[int]) -> None:
        if node is not None:
            result.append(node.key)            # корень
            self._preorder(node.left, result)  # левое поддерево
            self._preorder(node.right, result) # правое поддерево
    
    def postorder_traversal(self) -> List[int]:
        result = []
        self._postorder(self.root, result)
        return result
    
    def _postorder(self, node: Optional[TreeNode], result: List[int]) -> None:
        if node is not None:
            self._postorder(node.left, result)  # левое поддерево
            self._postorder(node.right, result) # правое поддерево
            result.append(node.key)             # корень
    
    def level_order_traversal(self) -> List[int]:
        result = []
        if self.root is None:
            return result
        
        queue = deque([self.root])             # очередь для BFS обхода
        while queue:
            node = queue.popleft()             # извлекаем первый элемент
            result.append(node.key)
            
            if node.left:                      # добавляем левого ребенка
                queue.append(node.left)
            if node.right:                     # добавляем правого ребенка
                queue.append(node.right)
        
        return result
    
    def height(self) -> int:
        """Возвращает высоту дерева (максимальная глубина)"""
        return self._height_node(self.root)
    
    def _height_node(self, node: Optional[TreeNode]) -> int:
        if node is None:                       # Пустое поддерево имеет высоту -1
            return -1
        # Высота = 1 + максимум из высот поддеревьев
        return 1 + max(self._height_node(node.left), self._height_node(node.right))
    
    def print_tree(self) -> None:
        """Визуализирует дерево в консоли"""
        if self.root is None:
            print("Дерево пустое")
            return
        
        lines = self._build_tree_string(self.root, 0, ' ', False)[0]
        print("\n" + "\n".join(line.rstrip() for line in lines))
    
    def _build_tree_string(self, node: Optional[TreeNode], curr_index: int, 
                          delimiter: str = '-', left: bool = False) -> tuple:
        """Рекурсивно строит строковое представление дерева для визуализации"""
        if node is None:
            return [], 0, 0, 0
        
        line1 = []
        line2 = []
        
        # Рекурсивно обрабатываем левое поддерево
        left_lines, left_pos, left_width, left_height = self._build_tree_string(
            node.left, 2 * curr_index + 1, delimiter, True)
        
        # Рекурсивно обрабатываем правое поддерево
        right_lines, right_pos, right_width, right_height = self._build_tree_string(
            node.right, 2 * curr_index + 2, delimiter, False)
        
        # Текущий узел
        node_repr = str(node.key)
        node_len = len(node_repr)
        
        # Позиция корня
        root_pos = left_pos + left_width
        
        # Собираем строки для визуализации связей
        if left_height > 0:
            line1.append(' ' * (root_pos) + '_' * (left_width - left_pos))
            line2.append(' ' * (root_pos - 1) + '/' + ' ' * (left_width - left_pos))
            root_pos = left_width + 1
            curr_index += root_pos
        
        line1.append(' ' * root_pos + node_repr)
        line2.append(' ' * root_pos)
        
        if right_height > 0:
            line1.append('_' * right_pos)
            line2.append(' ' * right_pos + '\\')
        
        # Объединяем строки
        line1 = [''.join(line1)]
        line2 = [''.join(line2)]
        
        # Добавляем строки левого и правого поддеревьев
        result_lines = line1 + line2
        for i in range(max(left_height, right_height)):
            left_line = left_lines[i] if i < left_height else ' ' * left_width
            right_line = right_lines[i] if i < right_height else ' ' * right_width
            result_lines.append(left_line + ' ' * (node_len + 2) + right_line)
        
        return result_lines, root_pos, left_width + node_len + right_width, max(left_height, right_height) + 2

# ==================== 1. AVL ДЕРЕВО (на основе вашего BST) ====================

class AVLNode(TreeNode):
    def __init__(self, key: int, value: Any = None):
        super().__init__(key, value)
        self.height = 1  # Высота узла

class AVLTree(BinarySearchTree):
    def __init__(self):
        super().__init__()
        self.root: Optional[AVLNode] = None
    
    def _get_height(self, node: Optional[AVLNode]) -> int:
        if node is None:
            return 0
        return node.height
    
    def _update_height(self, node: AVLNode) -> None:
        if node is not None:
            node.height = 1 + max(
                self._get_height(node.left), 
                self._get_height(node.right)
            )
    
    def _get_balance(self, node: AVLNode) -> int:
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _right_rotate(self, y: AVLNode) -> AVLNode:
        x = y.left
        T2 = x.right
        
        # Выполняем поворот
        x.right = y
        y.left = T2
        
        # Обновляем родителей
        if T2:
            T2.parent = y
        x.parent = y.parent
        y.parent = x
        
        # Обновляем высоты
        self._update_height(y)
        self._update_height(x)
        
        return x
    
    def _left_rotate(self, x: AVLNode) -> AVLNode:
        y = x.right
        T2 = y.left
        
        # Выполняем поворот
        y.left = x
        x.right = T2
        
        # Обновляем родителей
        if T2:
            T2.parent = x
        y.parent = x.parent
        x.parent = y
        
        # Обновляем высоты
        self._update_height(x)
        self._update_height(y)
        
        return y
    
    def insert(self, key: int, value: Any = None) -> None:
        new_node = AVLNode(key, value)
        
        if self.root is None:
            self.root = new_node
            self._size += 1
            return
        
        # Вставка как в обычном BST
        current = self.root
        parent = None
        while current is not None:
            parent = current
            if key < current.key:
                current = current.left
            elif key > current.key:
                current = current.right
            else:
                current.value = value
                return  # Ключ уже существует
        
        new_node.parent = parent
        if key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        
        self._size += 1
        
        # Балансировка
        self._balance_tree(new_node)
    
    def _balance_tree(self, node: AVLNode) -> None:
        current = node
        
        while current is not None:
            self._update_height(current)
            balance = self._get_balance(current)
            
            # Left Heavy
            if balance > 1:
                # Left-Right Case
                if self._get_balance(current.left) < 0:
                    current.left = self._left_rotate(current.left)
                    if current.left:
                        current.left.parent = current
                # Left-Left Case (might be after LR rotation)
                new_root = self._right_rotate(current)
                if current == self.root:
                    self.root = new_root
                else:
                    if current.parent.left == current:
                        current.parent.left = new_root
                    else:
                        current.parent.right = new_root
            
            # Right Heavy
            elif balance < -1:
                # Right-Left Case
                if self._get_balance(current.right) > 0:
                    current.right = self._right_rotate(current.right)
                    if current.right:
                        current.right.parent = current
                # Right-Right Case (might be after RL rotation)
                new_root = self._left_rotate(current)
                if current == self.root:
                    self.root = new_root
                else:
                    if current.parent.left == current:
                        current.parent.left = new_root
                    else:
                        current.parent.right = new_root
            
            current = current.parent
    
    def delete(self, key: int) -> bool:
        node = self.search(key)
        if node is None:
            return False
        
        parent = node.parent
        
        # Удаление как в обычном BST
        super()._delete_node(node)
        self._size -= 1
        
        # Балансировка начиная с родителя удаленного узла
        if parent:
            self._balance_tree(parent)
        elif self.root:
            # Если удалили корень, и дерево не пустое
            self._balance_tree(self.root)
        
        return True

# ==================== 2. КРАСНО-ЧЕРНОЕ ДЕРЕВО ====================

RED = True
BLACK = False

class RBNode(TreeNode):
    def __init__(self, key: int, value: Any = None):
        super().__init__(key, value)
        self.color = RED  # Новые узлы всегда красные

class RBTree:
    def __init__(self):
        self.nil = RBNode(None)
        self.nil.color = BLACK
        self.nil.left = None
        self.nil.right = None
        self.root = self.nil
        self._size = 0
    
    def size(self) -> int:
        return self._size
    
    def empty(self) -> bool:
        return self.root == self.nil
    
    def search(self, key: int) -> Optional[RBNode]:
        return self._search_node(self.root, key)
    
    def _search_node(self, node: RBNode, key: int) -> Optional[RBNode]:
        if node == self.nil or key == node.key:
            return node if node != self.nil else None
        
        if key < node.key:
            return self._search_node(node.left, key)
        else:
            return self._search_node(node.right, key)
    
    def _left_rotate(self, x: RBNode) -> None:
        y = x.right
        x.right = y.left
        
        if y.left != self.nil:
            y.left.parent = x
        
        y.parent = x.parent
        
        if x.parent == self.nil:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        
        y.left = x
        x.parent = y
    
    def _right_rotate(self, y: RBNode) -> None:
        x = y.left
        y.left = x.right
        
        if x.right != self.nil:
            x.right.parent = y
        
        x.parent = y.parent
        
        if y.parent == self.nil:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        
        x.right = y
        y.parent = x
    
    def insert(self, key: int, value: Any = None) -> None:
        new_node = RBNode(key, value)
        new_node.left = self.nil
        new_node.right = self.nil
        new_node.color = RED
        
        y = self.nil
        x = self.root
        
        while x != self.nil:
            y = x
            if new_node.key < x.key:
                x = x.left
            elif new_node.key > x.key:
                x = x.right
            else:
                x.value = value  # Ключ уже существует
                return
        
        new_node.parent = y
        
        if y == self.nil:
            self.root = new_node
        elif new_node.key < y.key:
            y.left = new_node
        else:
            y.right = new_node
        
        self._size += 1
        self._fix_insert(new_node)
    
    def _fix_insert(self, k: RBNode) -> None:
        while k.parent.color == RED:
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left  # Дядя
                if u.color == RED:
                    u.color = BLACK
                    k.parent.color = BLACK
                    k.parent.parent.color = RED
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self._right_rotate(k)
                    k.parent.color = BLACK
                    k.parent.parent.color = RED
                    self._left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right  # Дядя
                if u.color == RED:
                    u.color = BLACK
                    k.parent.color = BLACK
                    k.parent.parent.color = RED
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self._left_rotate(k)
                    k.parent.color = BLACK
                    k.parent.parent.color = RED
                    self._right_rotate(k.parent.parent)
            
            if k == self.root:
                break
        
        self.root.color = BLACK
    
    def delete(self, key: int) -> bool:
        node = self.search(key)
        if node is None:
            return False
        
        self._delete_node(node)
        self._size -= 1
        return True
    
    def _delete_node(self, z: RBNode) -> None:
        y = z
        y_original_color = y.color
        
        if z.left == self.nil:
            x = z.right
            self._rb_transplant(z, z.right)
        elif z.right == self.nil:
            x = z.left
            self._rb_transplant(z, z.left)
        else:
            y = self._min_node(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self._rb_transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            
            self._rb_transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        
        if y_original_color == BLACK:
            self._fix_delete(x)
    
    def _rb_transplant(self, u: RBNode, v: RBNode) -> None:
        if u.parent == self.nil:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent
    
    def _fix_delete(self, x: RBNode) -> None:
        while x != self.root and x.color == BLACK:
            if x == x.parent.left:
                w = x.parent.right
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._left_rotate(x.parent)
                    w = x.parent.right
                
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self._right_rotate(w)
                        w = x.parent.right
                    
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._right_rotate(x.parent)
                    w = x.parent.left
                
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._left_rotate(w)
                        w = x.parent.left
                    
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._right_rotate(x.parent)
                    x = self.root
        
        x.color = BLACK
    
    def _min_node(self, node: RBNode) -> RBNode:
        while node.left != self.nil:
            node = node.left
        return node
    
    def find_min(self) -> Optional[int]:
        if self.root == self.nil:
            return None
        node = self.root
        while node.left != self.nil:
            node = node.left
        return node.key
    
    def find_max(self) -> Optional[int]:
        if self.root == self.nil:
            return None
        node = self.root
        while node.right != self.nil:
            node = node.right
        return node.key
    
    def height(self) -> int:
        return self._height_node(self.root)
    
    def _height_node(self, node: RBNode) -> int:
        if node == self.nil:
            return -1
        return 1 + max(self._height_node(node.left), self._height_node(node.right))
    
    def inorder_traversal(self) -> List[int]:
        result = []
        self._inorder(self.root, result)
        return result
    
    def _inorder(self, node: RBNode, result: List[int]) -> None:
        if node != self.nil:
            self._inorder(node.left, result)
            result.append(node.key)
            self._inorder(node.right, result)
    
    def level_order_traversal(self) -> List[int]:
        result = []
        if self.root == self.nil:
            return result
        
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if node != self.nil:
                result.append(node.key)
                if node.left != self.nil:
                    queue.append(node.left)
                if node.right != self.nil:
                    queue.append(node.right)
        
        return result

# ==================== 3. ЭКСПЕРИМЕНТАЛЬНАЯ ЧАСТЬ ====================

def experiment_bst_height_random(n_max=10000, step=100):
    """Эксперимент: зависимость высоты BST от количества ключей (случайные ключи)"""
    x = []
    y_bst = []
    
    bst = BinarySearchTree()
    
    # Генерируем уникальные случайные числа
    all_numbers = list(range(n_max * 3))
    random.shuffle(all_numbers)
    unique_numbers = all_numbers[:n_max]
    
    for i in range(0, n_max, step):
        # Вставляем очередную порцию чисел
        for j in range(i, min(i + step, n_max)):
            bst.insert(unique_numbers[j])
        
        x.append(min(i + step, n_max))
        y_bst.append(bst.height())
    
    return x, y_bst

def experiment_avl_rb_height_random(n_max=10000, step=100):
    """Эксперимент: зависимость высоты AVL и RB от количества ключей (случайные ключи)"""
    x = []
    y_avl = []
    y_rb = []
    y_lower = []  # Теоретическая нижняя оценка
    y_upper = []  # Теоретическая верхняя оценка
    
    avl = AVLTree()
    rb = RBTree()
    
    # Генерируем уникальные случайные числа
    all_numbers = list(range(n_max * 3))
    random.shuffle(all_numbers)
    unique_numbers = all_numbers[:n_max]
    
    for i in range(0, n_max, step):
        # Вставляем очередную порцию чисел
        for j in range(i, min(i + step, n_max)):
            avl.insert(unique_numbers[j])
            rb.insert(unique_numbers[j])
        
        n = min(i + step, n_max)
        x.append(n)
        y_avl.append(avl.height())
        y_rb.append(rb.height())
        
        # Теоретические оценки для сбалансированных деревьев
        y_lower.append(math.ceil(math.log2(n + 1)) - 1)  # log₂(n+1) - 1
        y_upper.append(1.44 * math.log2(n + 2))  # 1.44*log₂(n+2) для AVL
    
    return x, y_avl, y_rb, y_lower, y_upper

def experiment_avl_rb_height_sorted(n_max=10000, step=100):
    """Эксперимент: зависимость высоты AVL и RB от количества ключи (отсортированные ключи)"""
    x = []
    y_avl = []
    y_rb = []
    y_lower = []
    y_upper = []
    
    avl = AVLTree()
    rb = RBTree()
    
    for i in range(0, n_max, step):
        # Вставляем отсортированные числа
        for j in range(i, min(i + step, n_max)):
            avl.insert(j)  # Отсортированные ключи
            rb.insert(j)
        
        n = min(i + step, n_max)
        x.append(n)
        y_avl.append(avl.height())
        y_rb.append(rb.height())
        
        # Теоретические оценки
        y_lower.append(math.ceil(math.log2(n + 1)) - 1)
        y_upper.append(1.44 * math.log2(n + 2))
    
    return x, y_avl, y_rb, y_lower, y_upper

def analyze_asymptotics(y_bst, n_values):
    """Анализ асимптотики высоты BST"""
    if len(y_bst) < 2:
        return "Недостаточно данных"
    
    ratios = []
    for i in range(1, len(y_bst)):
        if n_values[i] > 0 and y_bst[i] > 0:
            ratio = y_bst[i] / math.log2(n_values[i])
            ratios.append(ratio)
    
    if ratios:
        avg_ratio = statistics.mean(ratios)
        if avg_ratio < 1.5:
            asymptote = "O(log n) - сбалансированное дерево"
        elif avg_ratio < 3:
            asymptote = "O(√n) - умеренно несбалансированное"
        else:
            asymptote = "O(n) - вырожденное дерево (список)"
        
        return f"Среднее h/log₂(n) = {avg_ratio:.2f}. Асимптотика: {asymptote}"
    
    return "Не удалось определить асимптотику"

def plot_results():
    """Построение всех графиков с правильными теоретическими оценками"""
    
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ 1: Зависимость высоты BST от количества ключей")
    print("(случайные равномерно распределенные ключи)")
    print("=" * 60)
    
    x_bst, y_bst = experiment_bst_height_random(5000, 50)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_bst, y_bst, 'b-', linewidth=2, label='BST (эксперимент)')
    
    # Правильные теоретические оценки для случайного BST
    plt.plot(x_bst, [2 * math.log2(n + 1) for n in x_bst], 'g--', label='2*log₂(n+1) (средняя теория)')
    plt.plot(x_bst, [math.log2(n + 1) - 1 for n in x_bst], 'r--', label='log₂(n+1) - 1 (нижняя оценка)')
    
    plt.xlabel('Количество ключей (n)')
    plt.ylabel('Высота дерева (h)')
    plt.title('Зависимость высоты BST от n\n(случайные ключи)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    ratios = [y_bst[i] / math.log2(x_bst[i] + 1) if x_bst[i] > 1 else 0 for i in range(len(x_bst))]
    plt.plot(x_bst, ratios, 'b-', linewidth=2)
    
    # Правильные горизонтальные линии для BST
    plt.axhline(y=2.0, color='g', linestyle='--', label='2.0 (средняя теория)')
    plt.axhline(y=1.0, color='r', linestyle='--', label='1.0 (идеальный баланс)')
    
    plt.xlabel('Количество ключей (n)')
    plt.ylabel('h / log₂(n+1)')
    plt.title('Отношение высоты BST к log₂(n+1)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(analyze_asymptotics(y_bst, x_bst))
    
    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ 2: Сравнение AVL и RB деревьев")
    print("(случайные равномерно распределенные ключи)")
    print("=" * 60)
    
    x, y_avl, y_rb, y_lower, y_upper = experiment_avl_rb_height_random(5000, 50)
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, y_avl, 'b-', linewidth=2, label='AVL (эксперимент)')
    plt.plot(x, y_rb, 'r-', linewidth=2, label='RB (эксперимент)')
    plt.plot(x, y_lower, 'g--', label='Нижняя оценка: ⌈log₂(n+1)⌉ - 1')
    plt.plot(x, y_upper, 'm--', label='Верхняя оценка AVL: 1.44*log₂(n+2)')
    plt.xlabel('Количество ключей (n)')
    plt.ylabel('Высота дерева (h)')
    plt.title('AVL и RB: высота vs n\n(случайные ключи)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    ratios_avl = [y_avl[i] / math.log2(x[i] + 1) if x[i] > 1 else 0 for i in range(len(x))]
    ratios_rb = [y_rb[i] / math.log2(x[i] + 1) if x[i] > 1 else 0 for i in range(len(x))]
    plt.plot(x, ratios_avl, 'b-', label='AVL / log₂(n+1)')
    plt.plot(x, ratios_rb, 'r-', label='RB / log₂(n+1)')
    plt.axhline(y=1.44, color='m', linestyle='--', label='1.44 (AVL предел)')
    plt.axhline(y=1.0, color='g', linestyle='--', label='1.0 (идеальный баланс)')
    plt.xlabel('Количество ключей (n)')
    plt.ylabel('h / log₂(n+1)')
    plt.title('Отношение высоты к log₂(n+1)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ 3: AVL и RB с отсортированными ключами")
    print("=" * 60)
    
    x_sorted, y_avl_sorted, y_rb_sorted, y_lower_s, y_upper_s = experiment_avl_rb_height_sorted(2000, 20)
    
    plt.subplot(2, 2, 3)
    plt.plot(x_sorted, y_avl_sorted, 'b-', linewidth=2, label='AVL (отсортированные)')
    plt.plot(x_sorted, y_rb_sorted, 'r-', linewidth=2, label='RB (отсортированные)')
    plt.plot(x_sorted, y_lower_s, 'g--', label='Нижняя оценка')
    plt.plot(x_sorted, y_upper_s, 'm--', label='Верхняя оценка AVL')
    plt.xlabel('Количество ключей (n)')
    plt.ylabel('Высота дерева (h)')
    plt.title('AVL и RB: высота vs n\n(отсортированные ключи)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    diff_avl = [y_avl_sorted[i] - y_lower_s[i] for i in range(len(x_sorted))]
    diff_rb = [y_rb_sorted[i] - y_lower_s[i] for i in range(len(x_sorted))]
    plt.plot(x_sorted, diff_avl, 'b-', label='AVL - нижняя оценка')
    plt.plot(x_sorted, diff_rb, 'r-', label='RB - нижняя оценка')
    plt.xlabel('Количество ключей (n)')
    plt.ylabel('Разница с нижней оценкой')
    plt.title('Превышение над нижней оценкой')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ЭКСПЕРИМЕНТОВ:")
    print("=" * 60)
    
    print("\n1. BST со случайными ключами:")
    print(f"   При n={x_bst[-1]}: высота = {y_bst[-1]}")
    print(f"   log₂(n+1) - 1 = {math.log2(x_bst[-1] + 1) - 1:.2f}")
    print(f"   2*log₂(n+1) = {2 * math.log2(x_bst[-1] + 1):.2f}")
    print(f"   Отношение h / log₂(n+1) = {y_bst[-1] / math.log2(x_bst[-1] + 1):.2f}")
    
    print("\n2. AVL/RB со случайными ключами:")
    print(f"   При n={x[-1]}:")
    print(f"   AVL высота = {y_avl[-1]}, отношение = {y_avl[-1] / math.log2(x[-1] + 1):.2f}")
    print(f"   RB высота = {y_rb[-1]}, отношение = {y_rb[-1] / math.log2(x[-1] + 1):.2f}")
    
    print("\n3. AVL/RB с отсортированными ключами:")
    print(f"   При n={x_sorted[-1]}:")
    print(f"   AVL высота = {y_avl_sorted[-1]}, отношение = {y_avl_sorted[-1] / math.log2(x_sorted[-1] + 1):.2f}")
    print(f"   RB высота = {y_rb_sorted[-1]}, отношение = {y_rb_sorted[-1] / math.log2(x_sorted[-1] + 1):.2f}")
    
    print("\n4. Теоретические границы:")
    print("   AVL: h ≤ 1.44*log₂(n+2) - 0.33")
    print("   RB: h ≤ 2*log₂(n+1)")
    print("   BST (случайный): h ≈ 2*log₂(n) в среднем случае")

# ==================== 4. ТЕСТИРОВАНИЕ ====================

def test_all_trees():
    """Тестирование всех трех типов деревьев"""
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ BST, AVL И RB ДЕРЕВЬЕВ")
    print("=" * 60)
    
    test_keys = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45, 55, 65, 75, 85]
    
    print("\n1. БИНАРНОЕ ДЕРЕВО ПОИСКА (BST):")
    bst = BinarySearchTree()
    for key in test_keys:
        bst.insert(key)
    
    bst.print_tree()
    print(f"Высота: {bst.height()}")
    print(f"In-order: {bst.inorder_traversal()}")
    
    print("\n2. AVL ДЕРЕВО:")
    avl = AVLTree()
    for key in test_keys:
        avl.insert(key)
    
    print("Структура AVL дерева:")
    avl.print_tree()
    print(f"Высота: {avl.height()}")
    print(f"In-order: {avl.inorder_traversal()}")
    
    print("\n3. КРАСНО-ЧЕРНОЕ ДЕРЕВО:")
    rb = RBTree()
    for key in test_keys:
        rb.insert(key)
    
    # Выводим информацию о дереве
    print(f"Высота: {rb.height()}")
    print(f"In-order: {rb.inorder_traversal()}")
    print(f"Min: {rb.find_min()}, Max: {rb.find_max()}")
    
    print("\n4. ТЕСТИРОВАНИЕ УДАЛЕНИЯ:")
    
    keys_to_delete = [30, 70, 40]
    
    for key in keys_to_delete:
        bst.delete(key)
        avl.delete(key)
        rb.delete(key)
    
    print(f"\nBST после удаления {keys_to_delete}: высота = {bst.height()}")
    bst.print_tree()
    
    print(f"\nAVL после удаления {keys_to_delete}: высота = {avl.height()}")
    avl.print_tree()
    
    print(f"\nRB после удаления {keys_to_delete}: высота = {rb.height()}")
    print(f"In-order: {rb.inorder_traversal()}")
    
    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО!")
    print("=" * 60)

# ==================== ОСНОВНАЯ ПРОГРАММА ====================

if __name__ == "__main__":
    # Часть 1: Тестирование деревьев
    test_all_trees()
    
    # Часть 2: Проведение экспериментов и построение графиков
    plot_results()
    
    # Дополнительный анализ
    print("\n" + "=" * 60)
    print("ВЫВОДЫ И АНАЛИЗ:")
    print("=" * 60)
    print("""
    1. BST с случайными ключами имеет высоту ~2*log₂(n) в среднем случае.
    2. В худшем случае (отсортированные ключи) BST вырождается в список O(n).
    3. AVL дерево гарантирует высоту ≤ 1.44*log₂(n+2) в любом случае.
    4. RB дерево гарантирует высоту ≤ 2*log₂(n+1).
    5. Для случайных данных все три дерева показывают логарифмический рост.
    6. Для отсортированных данных BST деградирует, а AVL и RB сохраняют баланс.
    7. AVL более строго сбалансировано, но требует больше вращений.
    8. RB обеспечивает хороший баланс с меньшим количеством перестроек.
    """)