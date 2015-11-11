package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/goml/gobrain"
	"github.com/gonum/plot"
	//"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
	"github.com/gonum/plot/vg/draw"
	"github.com/pointlander/kmeans"
	"github.com/skelterjohn/go.matrix"
)

const (
	NEURAL_WIDTH  = 4
	NEURAL_MIDDLE = 3
	MODEL_WIDTH   = (NEURAL_WIDTH+1)*(NEURAL_MIDDLE+1) + (NEURAL_MIDDLE+1)*NEURAL_WIDTH
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
				y := s*x + y2
				iter(i, x2.Add(time.Duration(x)*time.Hour), y)
				i++
			}
		}
		iter(i, x2, y2)
		i++
		x1, y1 = x2, y2
	}
}

type Quote struct {
	t     time.Time
	price float64
}

func norm(x *matrix.DenseMatrix) {
	rows, cols := x.Rows(), x.Cols()
	for i := 0; i < rows; i++ {
		sum := float64(0)
		for j := 0; j < cols; j++ {
			k := x.Get(i, j)
			sum += k * k
		}
		sum = math.Sqrt(sum)
		if sum == 0 {
			continue
		}
		for j := 0; j < cols; j++ {
			x.Set(i, j, x.Get(i, j)/sum)
		}
	}
}

func subtract(x *matrix.DenseMatrix, y *matrix.DenseMatrix) {
	rows, cols := x.Rows(), x.Cols()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			x.Set(i, j, x.Get(i, j)-y.Get(0, j))
		}
	}
}

func mean(x *matrix.DenseMatrix) *matrix.DenseMatrix {
	rows, cols := x.Rows(), x.Cols()
	m, n := matrix.Zeros(1, cols), 0
	for i := 0; i < rows; i++ {
		zero := true
		for j := 0; j < cols; j++ {
			xij := x.Get(i, j)
			if xij != 0 {
				zero = false
			}
			m.Set(0, j, m.Get(0, j)+xij)
		}
		/* zero vectors skew the mean preventing good centering*/
		if !zero {
			n++
		}
	}
	for j := 0; j < cols; j++ {
		m.Set(0, j, m.Get(0, j)/float64(n))
	}
	return m
}

func cov(x *matrix.DenseMatrix) *matrix.DenseMatrix {
	y := x.Transpose()
	z, err := y.TimesDense(x)
	if err != nil {
		log.Fatal(err)
	}
	N, rows, cols := float64(x.Rows()), z.Rows(), z.Cols()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			z.Set(i, j, z.Get(i, j)/N)
		}
	}
	return z
}

func PCA(m *matrix.DenseMatrix, size int) *matrix.DenseMatrix {
	//norm(m)
	subtract(m, mean(m))
	var v *matrix.DenseMatrix
	{
		_, _, V, err := cov(m).SVD()
		if err != nil {
			log.Fatal(err)
		}
		rows := V.Rows()
		v = matrix.Zeros(rows, size)
		for i := 0; i < rows; i++ {
			for j := 0; j < size; j++ {
				v.Set(i, j, V.Get(i, j))
			}
		}
	}
	r, err := m.TimesDense(v)
	if err != nil {
		log.Fatal(err)
	}
	return r
}

type CPoint struct {
	X, Y float64
	C    int
}

type CPoints struct {
	CPoints []CPoint
}

func (cp *CPoints) Plot(c draw.Canvas, plt *plot.Plot) {
	trX, trY := plt.Transforms(&c)
	for _, point := range cp.CPoints {
		x := trX(point.X)
		y := trY(point.Y)

		var p vg.Path
		p.Move(x+1, y)
		p.Arc(x, y, 1, 0, 2*math.Pi)
		p.Close()
		switch point.C {
		case 0:
			c.SetColor(color.RGBA{0, 255, 0, 255})
		case 1:
			c.SetColor(color.RGBA{255, 0, 0, 255})
		case 2:
			c.SetColor(color.RGBA{0, 0, 255, 255})
		}
		c.Fill(p)
	}
}

func (cp *CPoints) DataRange() (xmin, xmax, ymin, ymax float64) {
	xmin, xmax, ymin, ymax = math.MaxFloat64, 0, math.MaxFloat64, 0
	for _, point := range cp.CPoints {
		if point.X < xmin {
			xmin = point.X
		}
		if point.X > xmax {
			xmax = point.X
		}
		if point.Y < ymin {
			ymin = point.Y
		}
		if point.Y > ymax {
			ymax = point.Y
		}
	}
	return
}

func main() {
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
	minYear, maxYear, size := time.Now().Year()+1, 0, 0
	iterator(data, func(index int, t time.Time, price float64) {
		size++
		year := t.Year()
		if year < minYear {
			minYear = year
		}
		if year > maxYear {
			maxYear = year
		}
		//fmt.Printf("%v %v\n", t, price)
	})
	prices := make([]Quote, size)
	iterator(data, func(index int, t time.Time, price float64) {
		i := size - 1 - index
		prices[i].t = t
		prices[i].price = price
	})
	model := matrix.Zeros(maxYear-minYear+1, MODEL_WIDTH)
	train := func(year int, patterns [][][]float64) {
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
		rand.Seed(0)
		ff := &gobrain.FeedForward{}
		ff.Init(NEURAL_WIDTH, NEURAL_MIDDLE, NEURAL_WIDTH)
		ff.SetContexts(7, nil)
		ff.Train(patterns, 1, 0.6, 0.4, true)
		i, j := year-minYear, 0
		for _, weights := range ff.InputWeights {
			for _, weight := range weights {
				model.Set(i, j, weight)
				j++
			}
		}
		for _, weights := range ff.OutputWeights {
			for _, weight := range weights {
				model.Set(i, j, weight)
				j++
			}
		}
	}
	patterns, i := [][][]float64{}, 0
	for i+3*7+1 < size {
		if prices[i].t.Year() != prices[i+3*7+1].t.Year() {
			fmt.Println(prices[i].t)
			train(prices[i].t.Year(), patterns)
			patterns = [][][]float64{}
			i += 3*7 + 1
			if i+3*7+1 >= size {
				break
			}
		}
		in, out := make([]float64, 4), make([]float64, 4)
		for j := 0; j < 4; j++ {
			in[j] = prices[i+j*7].price
			out[j] = prices[i+j*7+1].price
		}
		patterns = append(patterns, [][]float64{in, out})
		i++
	}
	fmt.Println(prices[i-1].t)
	train(prices[i-1].t.Year(), patterns)

	//norm(model)
	rows, cols := model.Rows(), model.Cols()
	rawData := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		rawData[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			rawData[i][j] = model.Get(i, j)
		}
	}
	labels, _, err := kmeans.Kmeans(rawData, 3, kmeans.CanberraDistance, 10)
	if err != nil {
		log.Fatal(err)
	}

	r := PCA(model.Copy(), 2)
	rows = r.Rows()
	points := make([]CPoint, rows)
	for i := 0; i < rows; i++ {
		points[i].X, points[i].Y, points[i].C =
			r.Get(i, 0), r.Get(i, 1), labels[i]
	}

	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Title.Text = "Market Dynamics"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(&CPoints{points})
	if err := p.Save(512, 512, "market_dynamics.png"); err != nil {
		log.Fatal(err)
	}

	ticks := []plot.Tick{}
	for _, price := range prices {
		if price.t.Year()%2 == 0 && price.t.YearDay() == 1 {
			ticks = append(ticks, plot.Tick{float64(price.t.Unix()), fmt.Sprintf("%v", price.t.Year())})
		}
	}

	cpoints := make([]CPoint, len(prices))
	for i, price := range prices {
		cpoints[i].X, cpoints[i].Y, cpoints[i].C =
			float64(price.t.Unix()), price.price, labels[price.t.Year()-minYear]
	}

	p, err = plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Title.Text = "DJIA"
	p.X.Label.Text = "Time"
	p.Y.Label.Text = "Price"
	p.X.Tick.Marker = plot.ConstantTicks(ticks)
	p.Add(&CPoints{cpoints})
	if err := p.Save(2048, 2048, "market_prices.png"); err != nil {
		log.Fatal(err)
	}
}
