package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/goml/gobrain"
)

func convert(item []string) (time.Time, float64) {
	t, err := time.Parse("2006-01-02", item[0])
	if err != nil {
		log.Fatal(err)
	}
	p, err := strconv.ParseFloat(item[1], 64)
	if err != nil {
		log.Fatal(err)
	}
	return t, p
}

func iterator(items [][]string, iter func(index int, t time.Time, price float64)) {
	i := 0
	x1, y1 := convert(items[0])
	iter(i, x1, y1)
	i++
	for _, item := range items[1:] {
		x2, y2 := convert(item)
		hours := x1.Sub(x2).Hours()
		if hours > 24 {
			s := (y1 - y2) / hours
			for x := hours - 24; x > 0; x -= 24 {
				y := s * x + y2
				iter(i, x2.Add(time.Duration(x) * time.Hour), y)
				i++
			}
		}
		iter(i, x2, y2)
		i++
		x1, y1 = x2, y2
	}
}

type Quote struct {
	t time.Time
	price float64
}

func main() {
	rand.Seed(0)

	in, err := os.Open("UDJIAD1.csv")
	if err != nil {
		log.Fatal(err)
	}
	reader := csv.NewReader(in)
	data, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	data = data[1:]
	size := 0
	iterator(data, func(index int, t time.Time, price float64) {
		size++
		//fmt.Printf("%v %v\n", t, price)
	})
	prices := make([]Quote, size)
	iterator(data, func(index int, t time.Time, price float64) {
		i := size - 1 - index
		prices[i].t = t
		prices[i].price = price
	})
	train := func(patterns [][][]float64) {
		min, max := math.MaxFloat64, float64(0)
		for _, pattern := range patterns {
			for _, a := range pattern {
				for _, b := range a {
					if b < min {
						min = b
					}
					if b > max {
						max = b
					}
				}
			}
		}
		d := max - min
		for _, pattern := range patterns {
			for _, a := range pattern {
				for b, _ := range a {
					a[b] = (a[b] - min) / d
				}
			}
		}
		ff := &gobrain.FeedForward{}
		ff.Init(4, 3, 4)
		ff.SetContexts(7, nil)
		ff.Train(patterns, 1, 0.6, 0.4, true)
	}
	patterns, i := [][][]float64{}, 0
	for i + 3 * 7 + 1 < size {
		if prices[i].t.Year() != prices[i + 3 * 7 + 1].t.Year() {
			fmt.Println(prices[i].t)
			train(patterns)
			patterns = [][][]float64{}
			i += 3 * 7 + 1
			if i + 3 * 7 + 1 >= size {
				break
			}
		}
		in, out := make([]float64, 4), make([]float64, 4)
		for j := 0; j < 4; j++ {
			in[j] = prices[i + j * 7].price
			out[j] = prices[i + j * 7 + 1].price
		}
		patterns = append(patterns, [][]float64{in, out})
		i++
	}
	fmt.Println(prices[i - 1].t)
	train(patterns)
}
