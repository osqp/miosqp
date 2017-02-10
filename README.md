# miOSQP


## TODO

-   [x] Add functions: add_left and add_right for each node
    -   [x] N.B. Update node l_i and u_i by restricting newly introduced ones in branching
-   [x] Store solution when upper bound improves (only x, dual does not matter)
-   [x] Add function to deal separately with root node (when branching)
-   [x] Complete get_bounds function
-   [x] Add proper warm starting (from parent node, when branching)
-   [x] Add only workspace passed to the nodes!
