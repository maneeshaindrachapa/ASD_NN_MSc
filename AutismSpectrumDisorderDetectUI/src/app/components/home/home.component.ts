import { Component, HostListener, OnInit } from '@angular/core';
import { faBrain, faInfoCircle } from '@fortawesome/free-solid-svg-icons';
import { UploadFilesService } from 'src/app/services/upload-files.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {
  faInfoCircle = faInfoCircle;
  faBrain = faBrain;
  selectedValue = 4;
  predictedValue = 'Positive'
  predictionDone = false;
  uploadCompleted = ""
  positive =""
  negative=""

  @HostListener('mousemove', ['$event'])
  public onAnything($event): void {
    this.uploadCompleted = localStorage.getItem("completed");
  }
  constructor(private uploadService: UploadFilesService) {
  }

  ngOnInit(): void {
    this.uploadCompleted = localStorage.getItem("completed");
  }

  selectChange(event) {
    console.log(this.selectedValue);
  }

  predict() {
    this.predictionDone = false;
    let modelName = "conv";
    if (this.selectedValue == 1) {
      modelName = "lstm";
    } else if (this.selectedValue == 2) {
      modelName = 'bilstm';
    } else if (this.selectedValue == 3) {
      modelName = 'gru';
    } else if (this.selectedValue == 4) {
      modelName = 'convlstm';
    } else if (this.selectedValue == 5) {
      modelName = 'caps';
    }
    let fileNames = localStorage.getItem('files');
    let participant_id = ((fileNames.split(","))[0].split("_"))[0];
    console.log(participant_id);
    this.uploadService.predict(modelName, participant_id).subscribe((data) => {
      if (data['asd'] == '1') {
        this.predictedValue = "Positive";
      } else {
        this.predictedValue = "Negative";
      }
      this.positive = ((data['positive']).toString()).substring(0,5);
      this.negative=((data['negative']).toString()).substring(0,5);

      this.predictionDone = true;
    }, (err) => {
      console.log(err);
    })
  }
}
