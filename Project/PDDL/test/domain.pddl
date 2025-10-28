(define (domain gripper-strips)
    (:requirements :negative-preconditions)
    (:predicates (room ?r)
                (ball ?b)
                (at ?b ?r)
                (at-robby ?r)
                (gripper ?g)
                (free ?g)
                (carry ?b ?g))
    (:action pick
        :parameters(?obj ?room ?gripper)
        :precondition 
            (and 
                (ball ?obj) 
                (room ?room) 
                (gripper ?gripper)
                (free ?gripper) 
                (at ?obj ?room) 
                (at-robby ?room)
            )
        :effect 
            (and 
                (not 
                    (free ?gripper)
                ) 
                (carry ?obj ?gripper) 
                (not (at ?obj ?room))
            )
        )
    (:action move
        :parameters (?from ?to)
        :precondition 
            (and 
                (room ?from) 
                (room ?to) 
                (at-robby ?from)
            )
        :effect 
            (and 
                (at-robby ?to)
                (not 
                    (at-robby ?from)
                )
            )
    )
    (:action drop
        :parameters (?room ?ball ?gripper)
        :precondition
            (and 
                (room ?room) 
                (ball ?ball) 
                (at-robby ?room) 
                (gripper ?gripper)
                (not 
                    (free ?gripper)
                ) 
                (carry ?ball ?gripper)
                (at-robby ?room))
        :effect 
            (and 
                (free ?gripper) 
                (at ?ball ?room) 
                (not
                    (carry ?ball ?gripper)
                )
            )
        )
)
