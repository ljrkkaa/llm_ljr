(define (problem letseat-sequence)
  (:domain letseat_plus)
  (:requirements :negative-preconditions)

  (:objects
    arm - robot
    cupcake - cupcake
    newcupcake - cupcake
    unicorn - unicorn
    table - location
    plate - location
  )

  (:init
    ; 机械臂和蛋糕初始位置
    (on arm table)
    (on cupcake table)
    (on newcupcake table)
    (on unicorn plate)

    ; 机械臂为空，独角兽饿
    (arm-empty)
    (uni-hungry)

    ; 移动路径
    (path table plate)
    (path plate table)
  )

  (:goal
    (and
      (on newcupcake plate)
      (eating unicorn cupcake)
      ; 独角兽不再饿
      (not (uni-hungry))
    )
  )
)