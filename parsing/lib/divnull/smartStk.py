import sys

stack = ['as', ',-_', 'survey']
arcs=[["survey", "LEFT-ARC(NMOD)", "October"], ["survey", "LEFT-ARC(NMOD)", "The"],
 ["managers", "LEFT-ARC(NMOD)", "purchasing"], ["managers", "LEFT-ARC(NMOD)", "corporate"],
 ["of", "RIGHT-ARC(PMOD)", "managers"], ["survey", "RIGHT-ARC(NMOD)", "of"],
 ["as", "RIGHT-ARC(SUB)", "expected"]]


def dfs(h,arcs,result):
    # 停止条件
    if not h in [arc[0] for arc in arcs]:
        return
    # あるheadについて、arcsを全走査
    for arc in arcs:
        if arc[0] == h:
            result.append(arc)
            dfs(arc[2],arcs,result)
    return result


if __name__ == '__main__':
    ans = dfs(stack[-1],arcs,[])
    print(ans)
