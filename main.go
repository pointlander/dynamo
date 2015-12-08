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

type Vector [MODEL_WIDTH]float64

/* https://en.wikipedia.org/wiki/Cosine_similarity */
func (a *Vector) similarity(b *Vector) float64 {
	cols := len(a)
	dot, maga, magb := float64(0), float64(0), float64(0)
	for j := 0; j < cols; j++ {
		aij, bij := a[j], b[j]
		dot += aij * bij
		maga += aij * aij
		magb += bij * bij
	}
	mag := math.Sqrt(maga * magb)
	if mag == 0 {
		return 0
	}
	return dot / mag
}

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

/* http://nghiaho.com/?page_id=1030 */
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
	R, B byte
}

type CPoints struct {
	CPoints []CPoint
	line    bool
}

func (cp *CPoints) Plot(c draw.Canvas, plt *plot.Plot) {
	trX, trY := plt.Transforms(&c)
	var lastX, lastY vg.Length
	for i, point := range cp.CPoints {
		x := trX(point.X)
		y := trY(point.Y)

		var p vg.Path
		p.Move(x+1, y)
		p.Arc(x, y, 1, 0, 2*math.Pi)
		p.Close()
		if point.C < 0 {
			c.SetColor(color.RGBA{0, point.R, point.B, 255})
		} else {
			switch point.C {
			case 0:
				c.SetColor(color.RGBA{0, 255, 0, 255})
			case 1:
				c.SetColor(color.RGBA{255, 0, 0, 255})
			case 2:
				c.SetColor(color.RGBA{0, 0, 255, 255})
			}
		}
		c.Fill(p)
		if cp.line && i > 0 {
			var p vg.Path
			p.Move(lastX, lastY)
			p.Line(x, y)
			c.Stroke(p)
		}
		lastX, lastY = x, y
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
	for i+3*7 < size {
		if prices[i].t.Year() != prices[i+3*7].t.Year() {
			fmt.Println(prices[i].t)
			train(prices[i].t.Year(), patterns)
			patterns = [][][]float64{}
			i += 3 * 7
			if i+3*7 >= size {
				break
			}
		}
		in, out := make([]float64, 4), make([]float64, 4)
		for j := 0; j < 4; j++ {
			in[j] = prices[i+j*7].price
			out[j] = prices[i+j*7].price
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
	marketDynamics := &CPoints{points, false}

	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Title.Text = "Market Dynamics"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(marketDynamics)
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
			float64(price.t.Unix()), price.price, -1 //labels[price.t.Year()-minYear]
	}
	marketPrices := &CPoints{cpoints, false}
	xmin, xmax, ymin, ymax := marketDynamics.DataRange()
	for i, price := range prices {
		point := points[price.t.Year()-minYear]
		cpoints[i].R = byte(255 * (point.X - xmin) / (xmax - xmin))
		cpoints[i].B = byte(255 * (point.Y - ymin) / (ymax - ymin))
	}

	p, err = plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Title.Text = "DJIA"
	p.X.Label.Text = "Time"
	p.Y.Label.Text = "Price"
	p.X.Tick.Marker = plot.ConstantTicks(ticks)
	p.Add(marketPrices)
	if err := p.Save(2048, 2048, "market_prices.png"); err != nil {
		log.Fatal(err)
	}

	var markov [3][3]int
	last := 0
	for _, label := range labels {
		markov[last][label]++
		last = label
	}
	colors := [...]string{"Green", "Red", "Blue"}
	for j, a := range markov {
		sum := 0.0
		for _, b := range a {
			sum += float64(b)
		}
		fmt.Println(colors[j])
		for i, b := range a {
			fmt.Printf("%.1f %v\n", 100*float64(b)/sum, colors[i])
		}
		fmt.Printf("\n")
	}

	rows, cols = model.Rows(), model.Cols()
	angles := make([]float64, rows)
	for i := 1; i < rows; i++ {
		a, b := &Vector{}, &Vector{}
		for j := 0; j < cols; j++ {
			a[j], b[j] = model.Get(i, j), model.Get(i-1, j)
		}
		angles[i] = a.similarity(b)
	}

	points = make([]CPoint, rows)
	for i := 0; i < rows; i++ {
		unix := time.Date(minYear+i, time.January, 0, 0, 0, 0, 0, time.UTC).Unix()
		points[i].X, points[i].Y, points[i].C =
			float64(unix), angles[i], labels[i]
	}
	marketSimilarity := &CPoints{points, true}

	p, err = plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Title.Text = "Market Similarity"
	p.X.Label.Text = "Time"
	p.Y.Label.Text = "Similarity"
	p.X.Tick.Marker = plot.ConstantTicks(ticks)
	p.Add(marketSimilarity)
	if err := p.Save(2048, 2048, "market_similarity.png"); err != nil {
		log.Fatal(err)
	}
}
