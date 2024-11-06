from py_wikiracer.internet import Internet
from typing import List
from html.parser import HTMLParser
from collections import deque, defaultdict
from heapq import heapify, heappush, heappop

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k,v in attrs:
                if k == "href":
                    self.urls.append(v)


class Parser:
    @staticmethod
    def get_links_in_page(html: str) -> List[str]:
        links = list()
        disallowed = Internet.DISALLOWED
        parser = MyHTMLParser()
        parser.urls = list()
        parser.feed(html)
        urls = parser.urls
        for url in urls:
            if "/wiki/" in url:
                if len(set(url.replace("/wiki/","")).intersection(set(disallowed))) == 0:
                    links.append(url)
        links = list(dict.fromkeys(links))
        return links


class BFSProblem:
    def __init__(self, internet: Internet):
        self.internet = internet
        self.visited = set()
        self.queue = deque()

    def bfs(self, source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia"):
        path = [source]
        self.queue = [(source, path)]
        if source == goal:
            self.internet.get_page(source)
            return path + [source]
        while self.queue:
            (vertex, path) = self.queue.pop(0)
            self.visited.add(vertex)
            html = self.internet.get_page(vertex)
            vertex_links = Parser.get_links_in_page(html)
            for next in vertex_links:
                if next not in self.visited:
                    if next == goal:
                        return path + [next]
                    else:
                        self.queue.append((next, path+[next]))
        return None


class DFSProblem:
    def __init__(self, internet: Internet):
        self.internet = internet
        self.visited = set()
        self.stack = deque()

    def dfs(self, source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia"):
        path = [source]
        self.stack = [(source, path)]
        if source == goal:
            self.internet.get_page(source)
            return path + [source]
        while self.stack:
            (vertex, path) = self.stack.pop()
            self.visited.add(vertex)
            html = self.internet.get_page(vertex)
            vertex_links = Parser.get_links_in_page(html)
            for next in vertex_links:
                if next not in self.visited:
                    if next == goal:
                        return path + [next]
                    else:
                        self.stack.append((next, path + [next]))
        return None


class DijkstrasProblem:
    def __init__(self, internet: Internet):
        self.internet = internet
        self.visited = set()
        self.heap = list()

    def dijkstras(self, source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia", costFn = lambda x, y: len(y)):
        heapify(self.heap)
        path = [source]
        heappush(self.heap, (0, source, path))
        lowest_cost = {source:0}
        if source == goal:
            self.internet.get_page(source)
            return path + [source]
        while self.heap:
            (cost, vertex, path) = heappop(self.heap)
            self.visited.add(vertex)
            html = self.internet.get_page(vertex)
            vertex_neighbors = Parser.get_links_in_page(html)
            for neighbor in vertex_neighbors:
                if neighbor not in self.visited:
                    neighbor_path = path + [neighbor]
                    if neighbor == goal:
                        return neighbor_path
                    if neighbor not in lowest_cost.keys():
                        lowest_cost[neighbor] = float("inf")
                    neighbor_cost = cost + costFn(vertex, neighbor)
                    if neighbor_cost < lowest_cost[neighbor]:
                        lowest_cost[neighbor] = neighbor_cost
                        heappush(self.heap, (lowest_cost[neighbor], neighbor, neighbor_path))
                else:
                    continue
        return None


class WikiracerProblem:
    def __init__(self, internet: Internet):
        self.internet = internet
        self.useless = {"A","a","An","an","the","The","of","for","in","on","Main","Page","to","and","from","by","ISBN"}
        self.useful = set()

    def _find_neighbors(self, vertex):
        if vertex == "random":
            html = self.internet.get_random()
        else:
            html = self.internet.get_page(vertex)
        links = Parser.get_links_in_page(html)
        return links

    def wikiracer(self, source = "/wiki/Calvin_Li", goal = "/wiki/Wikipedia"):
        f = open("py_wikiracer/wiki.txt", "r")
        for i in f:
            self.useful.add(i.strip())
        dij = DijkstrasProblem(self.internet)
        source_bfs_links = self._find_neighbors(source)
        source_bfs_links.append(source)
        if goal == source or goal in source_bfs_links:
            return [source] + [goal]
        goal_bfs_links = self._find_neighbors(goal)
        goal_bfs_links.append(goal)
        random_bfs_links = self._find_neighbors("random")
        common_links = set(random_bfs_links).intersection(set(source_bfs_links))
        self.goal_bfs2_links = list()
        goal_bfs_good_links = set()
        samples = set()
        list_of_lists = [i.replace("/wiki/","").replace("(","").replace(")","").split("_") for i in source_bfs_links]
        source_words = [val for sublist in list_of_lists for val in sublist]
        list_of_lists = [i.replace("/wiki/", "").replace("(", "").replace(")", "").split("_") for i in goal_bfs_links]
        goal_words = [val for sublist in list_of_lists for val in sublist]
        self.keywords = set(source_words).intersection(set(goal_words))
        for j in self.useless:
            self.keywords.discard(j)
        while len(goal_bfs_links) > 0:
            sample = goal_bfs_links.pop(0)
            if common_links is not None and sample not in common_links and sample != goal:
                sample_neighbors = self._find_neighbors(sample)
                self.goal_bfs2_links = self.goal_bfs2_links + sample_neighbors
                samples.add(sample)
                if goal in sample_neighbors:
                    goal_bfs_good_links.add(sample)
            self.x = set(source_bfs_links).intersection(set(self.goal_bfs2_links))
            if len(self.x) > 5 or source in (self.goal_bfs2_links + goal_bfs_links):
                break
            if len(goal_bfs_links) > 200 and len(self.goal_bfs2_links) + len(goal_bfs_links) + len(samples) > 5000:
                break
            if len(goal_bfs_links) > 50 and len(self.goal_bfs2_links) + len(goal_bfs_links) + len(samples) > 10000:
                break
        costFn = lambda node1, node2: (9999999999 if (node2 in common_links)
                                       else (0 if (node2 in goal_bfs_good_links)
                                             else (0.1 if (node2 in goal_bfs_links or node2 in samples)
                                                   else (1 if (node2 in self.useful or node2 in self.x)
                                                         else (10 if (node2 in self.goal_bfs2_links)
                                                               else (100 if (len(set(node2.replace("/wiki/","").replace("(", "").replace(")", "").split("_")).intersection(self.keywords))>0)
                                                                     else (1000 if (node2 in self.goal_bfs2_links)
                                                                           else 10000)))))))
        path = dij.dijkstras(source, goal, costFn)
        return path

# KARMA
class FindInPageProblem:
    def __init__(self, internet: Internet):
        self.internet = internet
    # This Karma problem is a little different. In this, we give you a source page, and then ask you to make up some heuristics that will allow you to efficiently
    #  find a page containing all of the words in `query`. Again, optimize for the fewest number of internet downloads, not for the shortest path.

    def find_in_page(self, source = "/wiki/Calvin_Li", query = ["ham", "cheese"]):

        raise NotImplementedError("Karma method find_in_page")

        path = [source]

        # find a path to a page that contains ALL of the words in query in any place within the page
        # path[-1] should be the page that fulfills the query.
        # YOUR CODE HERE

        return path # if no path exists, return None
