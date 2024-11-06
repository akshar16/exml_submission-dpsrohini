
function setToken(token) {
    localStorage.setItem('token', token);
    reloadSocketConnection();
}


function initializeSocket() {
   
    return io({
        extraHeaders: {
            'Authorization': 'Bearer ' + localStorage.getItem('token')
        }
    });
}

// Function to reload the socket connection
function reloadSocketConnection() {
    if (socket) {
        socket.disconnect();
    }
    socket = initializeSocket();
    setupSocketHandlers();
}

function setupSocketHandlers() {
    socket.on('connect', function () {
        console.log('Connected to server');
    });

    socket.on('board', (board) => {

        drawBoard(board[0]);
        show_points(board[1])
    })

    socket.on('player-connected', (name) => {
        player.innerHTML = name
    })
    socket.on('ghost-connected', (name) => {
        ghost.innerHTML = name
    })
    socket.on('token-refresh',()=>{
       
        location.reload()
    })


    socket.on('playerdocked', () => {
        player_dock.innerHTML = 'Docked'
    })
    socket.on('ghostdocked', () => {
        ghost_dock.innerHTML = 'Docked'
    })
    socket.on('undock', () => {
        player_dock.innerHTML = ''
        ghost_dock.innerHTML = ''
    })

    socket.on('token', (token) => {
        var player_token = token[0]
        var ghost_token = token[1]

        player.innerHTML = player_token
        ghost.innerHTML = ghost_token

    })

   


    socket.on('game-over', ({ winner, timestamps }) => {

        winnerdisplay = document.getElementById('winner')

        gameover.style.opacity = 1
        gameover.style.pointerEvents = 'all'
        document.querySelector('.gamearea').classList.add('overlay-active');

        if (winner == 'player') {
            winnerdisplay.innerHTML = 'Player'
        } else {
            winnerdisplay.innerHTML = 'Ghost'
        }

        player_timestamp = document.getElementById('player-timestamp')
        ghost_timestamp = document.getElementById('ghost-timestamp')

        player_timestamp.innerHTML = timestamps[0]
        ghost_timestamp.innerHTML = timestamps[1]
        
       
    })
    socket.on('score',(score)=>{
        scoreArea  = document.getElementById('score')
        scoreArea.innerHTML = score
  

    })
}

var player = document.getElementById('player-token')
var ghost = document.getElementById('ghost-token')

var player_dock = document.getElementById('player-dock')
var ghost_dock = document.getElementById('ghost-dock')

var gameover = document.getElementById('gameover')

var socket = initializeSocket()
setupSocketHandlers()
reloadSocketConnection();


document.addEventListener('DOMContentLoaded', function () {
    check_token(localStorage.getItem('token'))

    pacman = new Image();
    pacman.src = '/static/images/sprites/pacman.png';

    purple = new Image();
    purple.src = '/static/images/sprites/purple.png';

    red = new Image();
    red.src = '/static/images/sprites/red.png';

    reset_button = document.getElementById('reset')
    reset_button.addEventListener('click', reset)

    reset_button.style.display = 'none'
    end = document.getElementById('end')
    end.style.display = 'none'

    commands = document.getElementById('commands')
    commands.style.display = 'none'
 
    end.addEventListener('click', () => {
        socket.emit('end')
        this.location.reload()
    })

    var part_coll = document.getElementById('part-coll')
    var part_content = document.getElementById('part-content')

    part_coll.addEventListener('click', () => {

        if (part_content.style.display === 'block') {
            part_content.style.display = 'none'
            part_content.style.maxHeight = 0
        } else {
            part_content.style.display = 'block'
            part_content.style.maxHeight = part_content.scrollHeight + 'px'
        }
    })

    var moves_coll = document.getElementById('moves-coll')
    var moves_content = document.getElementById('moves-content')

    moves_coll.addEventListener('click', () => {
       
        if (moves_content.style.display === 'block') {
            moves_content.style.display = 'none'
            moves_content.style.maxHeight = 0
        } else {
            moves_content.style.display = 'block'
            moves_content.style.maxHeight = moves_content.scrollHeight + 'px'
        }
        
    })

    var close = document.getElementById('close-icon')

    close.addEventListener('click', () => {
        console.log('clicked')
        gameover.style.opacity = 0
        gameover.style.pointerEvents = 'none'
        document.querySelector('.gamearea').classList.remove('overlay-active');
    })


    queue = document.getElementById('queue-status')
    tokenInput = document.getElementById('token-input')
    

    saveTokenButton = document.getElementById('enter-queue');
    saveTokenButton.addEventListener('click', function () {
        var tokenInput = document.getElementById('token-input').value;
        enterToken(tokenInput);
        setToken(tokenInput);
        check_token(tokenInput)
        
    });

    emailInput = document.getElementById('email-input')
    emailButton = document.getElementById('get-token')

    emailButton.addEventListener('click',()=>{
        email = emailInput.value

      
        
        if (!validateEmail(email)){
            queue.innerHTML = 'Invalid Email'
            return console.log("invalid email")
        } 
        queue.innerHTML = "Sending Email..."
        getToken(email)
    })



})

const validateEmail = (email) => {
    return true;
}

const getToken = (email) => {
    fetch('/token/getToken',{
        method:"POST",
        headers:{
            'Content-Type':"application/json"
        },
        body: JSON.stringify({email:email})

    }).then(response=>{
        return response.json();
    }).then(data=>{
        if (data.error){
            return queue.innerHTML = data.error
        }
        if (data.status == "success"){
            return queue.innerHTML = "Your queue token has been sent to your email"
        }
    })
}

const enterToken = (token) => {
    fetch('/token/enterQueue',{
        method:"POST",
        headers:{
            'Content-Type':"application/json"
        },
        body: JSON.stringify({token:token})
    }).then(response=>{
        if (!response.ok) {
            return queue.innerHTML = 'Invalid Token'
        }
        return response.json();
    }).then(data=>{
        if (data.error){
            return queue.innerHTML = data.error
        }
        if (data.status == "success"){
            return queue.innerHTML = "You have been added to the queue"
        }
    }).catch(e=>{
        console.log(console.dire(e))
    })
}



const check_token = (token) => {
    fetch(`/token/get?token=${encodeURIComponent(token)}`,{
        method:"GET",
        headers:{
            'Content-Type':"application/json"
        }
    }).then(response=>{
        
        return response.json()
    }).then(data=>{
        if (!data.status){
            return queue.innerHTML = 'Your current token is expired/invalid'
        }
        if (data.status.role == 'player') {
            assign_player()

        }else if(data.status.role == 'observer'){
            assign_spectator(data.status.position)
        }
       
    }).catch(eror=>{

        console.error(eror)
    })
}


const assign_player = ()=>{
      queue.innerHTML = 'You are currently playing, find the player and ghost tokens above'
      tokenInput.style.display = 'none'
      saveTokenButton.style.display = 'none'
      reset_button.style.display = 'block'
      end.style.display = 'block'
      commands.style.display = 'block'
   
      emailInput.style.display = "none"
      emailButton.style.display = "none"
}

const assign_spectator = (position)=>{
    queue.innerHTML = `You are at position ${position} in the queue`
    tokenInput.style.display = 'none'
    saveTokenButton.style.display = 'none'
    reset_button.style.display = 'none'
    emailInput.style.display = "none"
      emailButton.style.display = "none"
}



const show_points = (points) => {
    var point = document.getElementById('player-points')
    var gameover_points = document.getElementById('score')
    point.innerHTML = points
    gameover_points.innerHTML = points
}

const reset = () => {
    socket.emit('reset')
    location.reload()
    


}



function setCanvasDimensions(x, y) {
    var canvas = document.getElementsByClassName('canvas-layer');
    for (var i = 0; i < canvas.length; i++) {
        canvas[i].width = x;
        canvas[i].height = y;
    }
}

function drawBoard(board) {
    var wallsCv = document.getElementById('walls');
    var wallsCtx = wallsCv.getContext('2d');
    var BackgroundCv = document.getElementById('background');
    var BackgroundCtx = BackgroundCv.getContext('2d');
    var foregroundCv = document.getElementById('foreground')
    var foregroundCtx = foregroundCv.getContext('2d'); 

    gameboard = document.getElementById('gameboard')
    gamestatus = document.getElementById('gamestatus')

    

    if (window.innerWidth > window.innerHeight) {
        var CellSize = (window.innerHeight / board.length)*70/100
        var x = board[0].length * CellSize
    var y = board.length * CellSize
         gamestatus.style.width = (window.innerWidth - x-100) + 'px'
    }else{
        var CellSize = (window.innerWidth / board[0].length)*90/100
        var x = board[0].length * CellSize
    var y = board.length * CellSize
    }
   
    

    gameboard.style.width = x + 'px'
    gameboard.style.height = y + 'px'

    setCanvasDimensions(gameboard.clientWidth,gameboard.clientHeight )

    BackgroundCtx.clearRect(0, 0, x, y);
    wallsCtx.clearRect(0, 0, x, y);
    foregroundCtx.clearRect(0, 0, x, y);

    for (var row = 0; row < board.length; row++) {
        for (var col = 0; col < board[0].length; col++) {

            try {
                cell = board[row][col]
            } catch {
               
                cell = board[row][col]

                continue
            }

            if (cell == "#") {
                wallsCtx.fillStyle = 'darkblue';
                wallsCtx.fillRect(col * CellSize, row * CellSize, CellSize, CellSize);
            }


            if (cell == ".") {
                foregroundCtx.fillStyle = 'orange';
                drawCircle(foregroundCtx, col * CellSize + CellSize / 2, row * CellSize + CellSize / 2, CellSize / 6);
            }
            if (cell == "p") {
             
                foregroundCtx.drawImage(pacman, col * CellSize, row * CellSize, CellSize, CellSize*0.9);
              
            }
            if (cell == "a") {
                foregroundCtx.drawImage(red, col * CellSize, row * CellSize, CellSize*0.9, CellSize*0.9);
            }
            if (cell == "b") {
                const offScreenCanvas = document.createElement('canvas');
                offScreenCanvas.width = CellSize;
                offScreenCanvas.height = CellSize;
                const offScreenCtx = offScreenCanvas.getContext('2d');
            
                // Apply filter to the off-screen canvas
                offScreenCtx.filter = 'hue-rotate(270deg)';
                offScreenCtx.drawImage(red, 0, 0, CellSize*0.9, CellSize*0.9);
            
                // Draw the filtered image onto the main canvas
                foregroundCtx.drawImage(offScreenCanvas, col * CellSize, row * CellSize, CellSize, CellSize);
            }
            if (cell == "c") {
                const offScreenCanvas = document.createElement('canvas');
                offScreenCanvas.width = CellSize;
                offScreenCanvas.height = CellSize;
                const offScreenCtx = offScreenCanvas.getContext('2d');
            
                // Apply filter to the off-screen canvas
                offScreenCtx.filter = 'hue-rotate(120deg)';
                offScreenCtx.drawImage(red, 0, 0, CellSize*0.9, CellSize*0.9);
            
                // Draw the filtered image onto the main canvas
                foregroundCtx.drawImage(offScreenCanvas, col * CellSize, row * CellSize, CellSize, CellSize);
            }
            if (cell == "d") {
                const offScreenCanvas = document.createElement('canvas');
                offScreenCanvas.width = CellSize;
                offScreenCanvas.height = CellSize;
                const offScreenCtx = offScreenCanvas.getContext('2d');
            
                // Apply filter to the off-screen canvas
                offScreenCtx.filter = 'hue-rotate(320deg)';
                offScreenCtx.drawImage(red, 0, 0, CellSize*0.9, CellSize*0.9);
            
                // Draw the filtered image onto the main canvas
                foregroundCtx.drawImage(offScreenCanvas, col * CellSize, row * CellSize, CellSize, CellSize);
            }
        }
    }
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    ctx.fill();
}
function drawCircle(ctx, x, y, radius) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.closePath();
    ctx.fill();
}






