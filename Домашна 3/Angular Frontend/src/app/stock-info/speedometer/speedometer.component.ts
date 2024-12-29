import { Component, Input } from '@angular/core';
import {NgStyle} from "@angular/common";

@Component({
    selector: 'app-speedometer',
    standalone: true,
    templateUrl: './speedometer.component.html',
    styleUrls: ['./speedometer.component.scss'],
    imports: [
        NgStyle
    ]
})
export class SpeedometerComponent {
    /** The gauge size (diameter) in px. */
    @Input() size = 200;
    /** Stroke width in px. */
    @Input() strokeWidth = 20;
    /** Current gauge value (optional). */
    @Input() value = 0;
    /** Maximum gauge value (optional). */
    @Input() maxValue = 100;
    /** Label to display in the center of the gauge. */
    @Input() label = 'Strong Sell';

    /**
     * Radius for our circles.
     */
    get radius(): number {
        return (this.size - this.strokeWidth) / 2;
    }

    /**
     * Full circumference if it were a complete circle
     */
    get circumference(): number {
        return 2 * Math.PI * this.radius;
    }

    /**
     * We'll use half the circumference for the semicircle.
     */
    get halfCircumference(): number {
        return this.circumference / 2;
    }

    /**
     * Maps [0..maxValue] to [0..halfCircumference] for the "fill"
     */
    get dashOffset(): number {
        // clamp value between 0 and maxValue
        const clamped = Math.max(0, Math.min(this.value, this.maxValue));
        const fillRatio = clamped / this.maxValue;
        // offset: (1 - fillRatio) * halfCircumference
        return this.halfCircumference * (1 - fillRatio);
    }

    /**
     * Helper method to compute transform for each label,
     * distributing them along the arc from left to right.
     *
     * For 5 labels, we want angles 180° → 0° in 4 even intervals.
     * i = 0 => angle = 180° (left)
     * i = 4 => angle = 0° (right)
     */
}