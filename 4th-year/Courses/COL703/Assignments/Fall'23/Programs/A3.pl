color(r).
color(p).
color(w).
color(b).
color(g).
cold_color(g).
cold_color(p).
neutral_color(b).
neutral_color(w).

% Check Red Box 
check_red(box(B1,B2)):- B1 \= p,
                        B2 \= p,
                        B1 \= r,
                        B2 \= r.

% Check Pink Box
check_pink(box(b,B2)) :- B2 \= p.  

% Check black Boxes
check_black(box(B1,B2)) :- cold_color(B1), 
                            cold_color(B2).  

% Check white boxes
check_white(box(r,g)).

% Check green boxes
check_green(box(B1,B2)) :- B1 \= g,
                            B2 \= g.

% Check all boxes
check_all(box(w,p),box(_,_), box(_,_),box(_,_),box(_,_)).
check_all(box(_,_), box(w,p), box(_,_), box(_,_), box(_,_)).
check_all(box(_,_), box(_,_), box(w,p), box(_,_), box(_,_)).
check_all(box(_,_), box(_,_), box(_,_), box(w,p), box(_,_)).
check_all(box(_,_), box(_,_), box(_,_), box(_,_), box(w,p)).

verify_boxes(BOX1,BOX2,BOX3,BOX4,BOX5) :- check_red(BOX1),
                                          check_pink(BOX2),
                                          check_white(BOX3),
                                          check_black(BOX4),
                                          check_green(BOX5),
                                          check_all(BOX1,BOX2,BOX3,BOX4,BOX5).

arrange([B1,B2,B3,B4,B5,B6,B7,B8,B9,B10],Y):- permutation([B1,B2,B3,B4,B5,B6,B7,B8,B9,B10],Y),
                verify_boxes(box(B1,B2),box(B3,B4),box(B5,B6),box(B7,B8),box(B9,B10)).

main :- arrange(Y,[r,r,p,p,w,w,b,b,g,g]),
        atomics_to_string(Y,S),
        upcase_atom(S,S1),
        write(S1),nl,
        halt.