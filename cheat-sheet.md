## cheat sheet

**无向图**
1.判断是否连通

```python
def is_connected(graph,n):
    visited = [False] * n
    stack = [0]
    visited[0] = True

    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if not visited[neighbor]:
                stack.append(neighbor)
                visited[neighbor] = True

    return all[visited]
```

2.判断是否有回路

```python
def cycle(graph,n):
    def dfs(node,visited,parent):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor,visited,node):
                    return True
            elif parent != neighbor:
                return True
        return False
    
    visited = [False] * n
    for node in range(n):
        if not visited[node]:
            if dfs(node,visited,-1):
                return True
    return False
```

heap
monotonous-stack
stack
deque
tree
单调栈