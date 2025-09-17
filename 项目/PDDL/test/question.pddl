(define (problem solve)
    (:domain gripper-strips)
    (:objects 
        rooma roomb ball1 ball2 left right
    )
    (:init 
        (room rooma)
        (room roomb) 
        (ball ball1) 
        (ball ball2)
        (gripper left) 
        (gripper right) 
        (free left) 
        (free right)
        (at ball1 rooma) 
        (at ball2 rooma) 
        (at-robby rooma)
    )
    (:goal 
        (and
            (at ball1 roomb)
            (at ball2 roomb)
        )
    )
)
