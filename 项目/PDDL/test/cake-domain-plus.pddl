(define (domain letseat_plus)

  (:requirements :typing)

  (:types
    location locatable - object
    bot cupcake uni - locatable
    robot - bot
    unicorn - uni
  )

  (:predicates
    (on ?obj - locatable ?loc - location)
    (holding ?arm - bot ?cupcake - locatable)
    (arm-empty)
    (path ?location1 - location ?location2 - location)
    (eating ?uni - uni ?cupcake - locatable)
    (uni-hungry)
  )

  (:action pick-up
    :parameters (?arm - bot ?cupcake - locatable ?loc - location)
    :precondition (and
      (on ?arm ?loc)
      (on ?cupcake ?loc)
      (arm-empty)
    )
    :effect (and
      (not (on ?cupcake ?loc))
      (holding ?arm ?cupcake)
      (not (arm-empty))
    )
  )

  (:action drop
    :parameters (?arm - bot ?cupcake - locatable ?loc - location)
    :precondition (and
      (on ?arm ?loc)
      (holding ?arm ?cupcake)
    )
    :effect (and
      (on ?cupcake ?loc)
      (arm-empty)
      (not (holding ?arm ?cupcake))
    )
  )

  (:action move
    :parameters (?arm - bot ?from - location ?to - location)
    :precondition (and
      (on ?arm ?from)
      (path ?from ?to)
    )
    :effect (and
      (not (on ?arm ?from))
      (on ?arm ?to)
    )
  )

  (:action eat
    :parameters (?unicorn - uni ?cupcake - locatable ?loc - location)
    :precondition (and
      (on ?unicorn ?loc)
      (on ?cupcake ?loc)
      (uni-hungry)
    )
    :effect (and
      (not (on ?cupcake ?loc))
      (eating ?unicorn ?cupcake)
      (not (uni-hungry))
    )
  )
)
