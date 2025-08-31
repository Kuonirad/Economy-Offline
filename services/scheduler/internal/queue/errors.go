package queue

import "errors"

var (
    ErrQueueFull          = errors.New("queue is at maximum capacity")
    ErrQueueEmpty         = errors.New("queue is empty")
    ErrItemNotFound       = errors.New("item not found in queue")
    ErrMaxAttemptsReached = errors.New("maximum retry attempts reached")
)